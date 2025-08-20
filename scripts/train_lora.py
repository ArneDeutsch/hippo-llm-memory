"""TRL + PEFT LoRA/QLoRA trainer (single GPU).

This script wires together Transformers, TRL's :class:`~trl.SFTTrainer` and
PEFT's :class:`~peft.LoraConfig` to provide a minimal LoRA/QLoRA finetuning
loop.  It is intentionally light‑weight and is only meant to exercise the code
paths in CI – real training will require a substantially more elaborate setup.

The configuration is handled through `hydra` so that users can override any of
the fields from the command line, e.g. ``python train_lora.py model_name=gpt2``.

The ``--dry-run``/``dry_run=true`` flag simply loads the model and tokenizer and
then exits.  This is used by the unit tests to ensure that the heavy training
loop does not run on CI.
"""

from __future__ import annotations

import inspect
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Ensure repo root is on the path when executed as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from hippo_mem.adapters.lora import (
    count_trainable_parameters,
    default_target_modules,
)
from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.adapter import RelationalAdapter
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.adapter import AdapterConfig as SpatialAdapterConfig
from hippo_mem.spatial.adapter import SpatialAdapter
from hippo_mem.spatial.map import PlaceGraph
from scripts.utils import log_memory_status

logging.basicConfig(level=logging.INFO)


@dataclass
class TrainConfig:
    """Configuration for LoRA/QLoRA training."""

    # Model & data
    model_name: str = field(
        default_factory=lambda: os.environ.get("HF_MODEL_PATH", "models/tiny-gpt2")
    )
    dataset_name: str = "imdb"
    data_format: str = "hf"  # {"hf","jsonl"}
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    output_dir: str = "outputs"

    # Run metadata
    seed: int = 0
    run_name: str = "default"
    notes: str = ""

    # Training hyper‑parameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 500
    learning_rate: float = 5e-5

    # LoRA specific parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=list)

    # Utility flags
    dry_run: bool = False

    # Episodic adapter configuration
    episodic: AdapterConfig = field(
        default_factory=lambda: AdapterConfig(
            hidden_size=16,
            num_heads=1,
            lora_r=16,
            lora_alpha=16,
            lora_dropout=0.1,
        )
    )

    # Relational and spatial memory knobs
    relational: bool = False
    spatial: SpatialAdapterConfig = field(
        default_factory=lambda: SpatialAdapterConfig(
            hidden_size=16,
            num_heads=1,
            lora_r=16,
            lora_alpha=16,
            lora_dropout=0.1,
        )
    )

    # Replay toggle
    @dataclass
    class Replay:
        enabled: bool = True

    replay: Replay = field(default_factory=Replay)

    # Batch mix for replay scheduling
    @dataclass
    class BatchMix:
        episodic: float = 0.5
        semantic: float = 0.3
        fresh: float = 0.2

    batch_mix: BatchMix = field(default_factory=BatchMix)

    @dataclass
    class Maintenance:
        maintenance_interval: float = 60.0
        decay_rate: float = 0.0
        prune_min_salience: float = 0.0
        prune_max_age: Optional[float] = None

    episodic_mem: Maintenance = field(default_factory=Maintenance)

    @dataclass
    class RelationalMaintenance:
        maintenance_interval: float = 300.0
        prune_min_conf: float = 0.0
        prune_max_age: Optional[float] = None

    relational_mem: RelationalMaintenance = field(default_factory=RelationalMaintenance)

    @dataclass
    class SpatialMaintenance:
        maintenance_interval: float = 100.0
        decay_rate: float = 0.0
        prune_max_age: Optional[int] = None

    spatial_mem: SpatialMaintenance = field(default_factory=SpatialMaintenance)

    @dataclass
    class Efficiency:
        flash_attention: bool = False
        mqa_gqa: Optional[str] = None

    efficiency: Efficiency = field(default_factory=Efficiency)


# Register the config with Hydra so that `@hydra.main` can locate it.
ConfigStore.instance().store(name="train_lora_config", node=TrainConfig)


def _load_model_and_tokenizer(cfg: TrainConfig):
    """Load model and tokenizer with 4‑bit quantisation when possible."""

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Ensure pad token is set for causal models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = {}
    try:  # bitsandbytes may be unavailable on some platforms
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            quant_kwargs["quantization_config"] = bnb_config
            quant_kwargs["device_map"] = "auto"
    except Exception:  # pragma: no cover - optional dependency
        pass

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **quant_kwargs)
    model.config.use_cache = False  # Needed for gradient checkpointing
    model.gradient_checkpointing_enable()
    if cfg.efficiency.flash_attention:
        try:
            model.set_attn_implementation("flash_attention_2")
        except Exception:  # pragma: no cover - optional dependency
            logging.warning("FlashAttention not available; using standard attention")
    return model, tokenizer


def _init_sft_trainer(
    model,
    dataset,
    training_args,
    tokenizer,
    peft_config,
    eval_dataset=None,
):
    """Instantiate :class:`~trl.SFTTrainer` handling API changes.

    Recent versions of TRL renamed the ``tokenizer`` argument to
    ``processing_class``.  This helper inspects the signature of
    :class:`~trl.SFTTrainer` at runtime and forwards the tokenizer using the
    appropriate parameter name to remain compatible across releases.
    """

    kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
        "peft_config": peft_config,
    }
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    sig = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    return SFTTrainer(**kwargs)


def train(cfg: TrainConfig) -> None:
    """Execute the training or dry‑run."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model, tokenizer = _load_model_and_tokenizer(cfg)

    hidden = getattr(model.config, "hidden_size", cfg.episodic.hidden_size)

    epi_adapter: EpisodicAdapter | None = None
    spat_adapter: SpatialAdapter | None = None

    # Determine KV head counts based on efficiency flag
    if cfg.efficiency.mqa_gqa == "mqa":
        epi_kv_heads = 1
        spat_kv_heads = 1
    elif cfg.efficiency.mqa_gqa == "gqa":
        epi_kv_heads = max(1, cfg.episodic.num_heads // 2)
        spat_kv_heads = max(1, cfg.spatial.num_heads // 2)
    else:
        epi_kv_heads = cfg.episodic.num_kv_heads or cfg.episodic.num_heads
        spat_kv_heads = cfg.spatial.num_kv_heads or cfg.spatial.num_heads

    if cfg.episodic.enabled:
        epi_cfg = AdapterConfig(
            hidden_size=hidden,
            num_heads=cfg.episodic.num_heads,
            num_kv_heads=epi_kv_heads,
            lora_r=cfg.episodic.lora_r,
            lora_alpha=cfg.episodic.lora_alpha,
            lora_dropout=cfg.episodic.lora_dropout,
            enabled=True,
            flash_attention=cfg.efficiency.flash_attention,
        )
        epi_adapter = EpisodicAdapter(epi_cfg)

    rel_adapter: RelationalAdapter | None = None
    if cfg.relational:
        rel_adapter = RelationalAdapter()

    if cfg.spatial.enabled:
        spat_cfg = SpatialAdapterConfig(
            hidden_size=hidden,
            num_heads=cfg.spatial.num_heads,
            num_kv_heads=spat_kv_heads,
            lora_r=cfg.spatial.lora_r,
            lora_alpha=cfg.spatial.lora_alpha,
            lora_dropout=cfg.spatial.lora_dropout,
            enabled=True,
        )
        spat_adapter = SpatialAdapter(spat_cfg)

    # Simulate interleaved replay batches using a scheduler
    store_cfg = {
        "hopfield": cfg.episodic.hopfield,
        "decay_rate": cfg.episodic_mem.decay_rate,
        "prune": {
            "min_salience": cfg.episodic_mem.prune_min_salience,
            "max_age": cfg.episodic_mem.prune_max_age,
        },
    }
    store = EpisodicStore(hidden, config=store_cfg)
    epi_interval = 0.1 if cfg.dry_run else cfg.episodic_mem.maintenance_interval
    store.start_background_tasks(epi_interval)

    kg_cfg = {
        "prune": {
            "min_conf": cfg.relational_mem.prune_min_conf,
            "max_age": cfg.relational_mem.prune_max_age,
        }
    }
    kg = KnowledgeGraph(config=kg_cfg)
    kg_interval = 0.1 if cfg.dry_run else cfg.relational_mem.maintenance_interval
    kg.start_background_tasks(kg_interval)

    spat_map_cfg = {
        "decay_rate": cfg.spatial_mem.decay_rate,
        "prune": {"max_age": cfg.spatial_mem.prune_max_age},
    }
    spatial_map = PlaceGraph(path_integration=cfg.spatial.enabled, config=spat_map_cfg)
    spat_interval = 0.1 if cfg.dry_run else cfg.spatial_mem.maintenance_interval
    spatial_map.start_background_tasks(spat_interval)

    scheduler = None
    worker = None
    if cfg.replay.enabled:
        scheduler = ReplayScheduler(store, kg, batch_mix=cfg.batch_mix)

        total = 10
        for i in range(total):
            # Add some dummy traces to the scheduler so episodic items can be sampled
            scheduler.add_trace(str(i), np.zeros(hidden, dtype=np.float32), score=float(i))

        # Launch background consolidation worker
        worker = ConsolidationWorker(
            scheduler,
            model,
            episodic_adapter=epi_adapter,
            relational_adapter=rel_adapter,
            spatial_adapter=spat_adapter,
        )
        worker.start()

    try:
        peft_config = None
        if isinstance(model, torch.nn.Module):
            modules = cfg.target_modules or default_target_modules(model)
            logging.info("Target modules: %s", modules)
            peft_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=modules,
            )

        train_ds = None
        val_ds = None
        if cfg.data_format == "jsonl":
            from scripts.jsonl_dataset import load_jsonl_files, split_train_val

            train_ds = load_jsonl_files(cfg.train_files)
            if cfg.val_files:
                val_ds = load_jsonl_files(cfg.val_files)
            else:
                train_ds, val_ds = split_train_val(cfg.train_files[0], None)
            logging.info("Train dataset size: %d", len(train_ds))
            logging.info("Validation dataset size: %d", len(val_ds))
        elif not cfg.dry_run:
            train_ds = load_dataset(cfg.dataset_name, split="train")
            logging.info("Train dataset size: %d", len(train_ds))

        if cfg.dry_run:
            if peft_config is not None:
                model = get_peft_model(model, peft_config)
                count = count_trainable_parameters(model)
                logging.info("Trainable parameters: %d", count)
                if count <= 0:
                    raise RuntimeError(
                        "No trainable parameters found. Set `target_modules` to attach LoRA.",
                    )
            for _ in range(3):
                log_memory_status(store, kg, spatial_map, scheduler, worker)
                time.sleep(0.1)
            return

        training_args = SFTConfig(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            logging_steps=1,
            bf16=True,
            gradient_checkpointing=True,
            dataset_text_field="text",
        )

        trainer = _init_sft_trainer(
            model,
            train_ds,
            training_args,
            tokenizer,
            peft_config,
            eval_dataset=val_ds,
        )
        count = count_trainable_parameters(trainer.model)
        logging.info("Trainable parameters: %d", count)
        if count <= 0:
            raise RuntimeError(
                "No trainable parameters found. Set `target_modules` to attach LoRA."
            )

        trainer.train()
    finally:
        if worker is not None:
            worker.stop()
            worker.join(timeout=1)


@hydra.main(config_name="train_lora_config", version_base=None)
def main(cfg: TrainConfig) -> None:  # pragma: no cover - thin wrapper
    """Hydra entry point."""

    train(cfg)


def parse_args(args: Optional[List[str]] = None) -> TrainConfig:
    """Parse a list of Hydra style overrides into a :class:`TrainConfig`."""

    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli(args or [])
    merged = OmegaConf.merge(cfg, cli_cfg)
    return OmegaConf.to_object(merged)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
