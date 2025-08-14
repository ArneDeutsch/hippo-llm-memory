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

import random
from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import numpy as np
import torch
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.relational.adapter import RelationalAdapter
from hippo_mem.spatial.adapter import AdapterConfig as SpatialAdapterConfig
from hippo_mem.spatial.adapter import SpatialAdapter


@dataclass
class TrainConfig:
    """Configuration for LoRA/QLoRA training."""

    # Model & data
    model_name: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "imdb"
    output_dir: str = "outputs"

    # Training hyper‑parameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 1
    learning_rate: float = 2e-4

    # LoRA specific parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Utility flags
    dry_run: bool = False

    # Episodic adapter configuration
    episodic: AdapterConfig = field(
        default_factory=lambda: AdapterConfig(hidden_size=16, num_heads=1)
    )

    # Relational and spatial memory knobs
    relational: bool = False
    spatial: SpatialAdapterConfig = field(
        default_factory=lambda: SpatialAdapterConfig(hidden_size=16, num_heads=1)
    )

    # Batch mix for replay scheduling
    @dataclass
    class BatchMix:
        episodic: float = 0.5
        semantic: float = 0.3
        fresh: float = 0.2

    batch_mix: BatchMix = field(default_factory=BatchMix)


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
    return model, tokenizer


def train(cfg: TrainConfig) -> None:
    """Execute the training or dry‑run."""

    model, tokenizer = _load_model_and_tokenizer(cfg)

    hidden = getattr(model.config, "hidden_size", cfg.episodic.hidden_size)

    epi_adapter: EpisodicAdapter | None = None
    if cfg.episodic.enabled:
        epi_cfg = AdapterConfig(
            hidden_size=hidden,
            num_heads=cfg.episodic.num_heads,
            num_kv_heads=cfg.episodic.num_kv_heads,
            lora_r=cfg.episodic.lora_r,
            lora_alpha=cfg.episodic.lora_alpha,
            lora_dropout=cfg.episodic.lora_dropout,
            enabled=True,
        )
        epi_adapter = EpisodicAdapter(epi_cfg)

    rel_adapter: RelationalAdapter | None = None
    if cfg.relational:
        rel_adapter = RelationalAdapter()

    spat_adapter: SpatialAdapter | None = None
    if cfg.spatial.enabled:
        spat_cfg = SpatialAdapterConfig(
            hidden_size=hidden,
            num_heads=cfg.spatial.num_heads,
            lora_r=cfg.spatial.lora_r,
            lora_alpha=cfg.spatial.lora_alpha,
            lora_dropout=cfg.spatial.lora_dropout,
            enabled=True,
        )
        spat_adapter = SpatialAdapter(spat_cfg)

    # Simulate interleaved replay batches (50/30/20 default mix)
    schedule = []
    total = 10
    schedule.extend(["episodic"] * int(cfg.batch_mix.episodic * total))
    schedule.extend(["semantic"] * int(cfg.batch_mix.semantic * total))
    remaining = total - len(schedule)
    schedule.extend(["fresh"] * remaining)
    random.shuffle(schedule)
    for kind in schedule:
        if kind == "episodic" and epi_adapter is not None:
            h = torch.zeros(1, 1, hidden)
            m = torch.zeros(1, 1, hidden)
            epi_adapter(h, m)
        elif kind == "semantic" and rel_adapter is not None:
            q = np.zeros(hidden, dtype=np.float32)
            k = np.zeros((1, hidden), dtype=np.float32)
            rel_adapter(q, k)
        elif kind == "fresh" and spat_adapter is not None:
            h = torch.zeros(1, 1, hidden)
            p = torch.zeros(1, 1, hidden)
            spat_adapter(h, p)

    # Short circuit for the unit tests / smoke runs
    if cfg.dry_run:
        return

    dataset = load_dataset(cfg.dataset_name, split="train")

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()


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
