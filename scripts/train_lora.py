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
from typing import Any, List, Optional

# Ensure repo root is on the path when executed as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from hippo_mem.adapters.episodic_adapter import EpisodicMemoryAdapter
from hippo_mem.adapters.lora import count_trainable_parameters, default_target_modules
from hippo_mem.adapters.patch import MemoryFusionConfig, attach_adapters
from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic import episodic_retrieve_and_pack
from hippo_mem.episodic.adapter import AdapterConfig
from hippo_mem.episodic.gating import WriteGate, gate_batch
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import AsyncStoreWriter, EpisodicStore, TraceValue
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.retrieval import relational_retrieve_and_pack
from hippo_mem.spatial.adapter import AdapterConfig as SpatialAdapterConfig
from hippo_mem.spatial.adapter import SpatialAdapter
from hippo_mem.spatial.map import PlaceGraph
from hippo_mem.spatial.retrieval import spatial_retrieve_and_pack
from scripts.utils import ingest_text, log_memory_status

# Backwards compatibility for tests expecting ``EpisodicAdapter`` symbol
EpisodicAdapter = EpisodicMemoryAdapter

logging.basicConfig(level=logging.INFO)


@dataclass
class EpisodicSpec:
    enabled: bool = False
    k: int = 0


@dataclass
class RelationalSpec:
    enabled: bool = False
    k: int = 0
    hops: int = 1
    schema_fasttrack: bool = True


@dataclass
class SpatialSpec:
    enabled: bool = False
    radius: int = 1
    max_nodes: int = 0
    max_edges: int = 0


@dataclass
class Memory:
    episodic: EpisodicSpec = field(default_factory=EpisodicSpec)
    relational: RelationalSpec = field(default_factory=RelationalSpec)
    spatial: SpatialSpec = field(default_factory=SpatialSpec)


@dataclass
class TrainConfig:
    """Configuration for LoRA/QLoRA training."""

    # Model & data
    model_name: str = field(
        default_factory=lambda: os.environ.get("HF_MODEL_PATH", "models/tiny-gpt2")
    )
    dataset_name: str = "imdb"
    data_format: str = "jsonl"  # {"hf","jsonl"}
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

    # Adapter fusion
    fusion_insert_block_index: int = -4

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

    write_threshold: float = 0.5

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

    memory: Memory = field(default_factory=Memory)

    # Replay toggle
    @dataclass
    class Replay:
        enabled: bool = False
        ratio: float = 0.3

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


def _gather_memory_tokens(
    hidden: torch.Tensor,
    cfg: TrainConfig,
    store: EpisodicStore | None,
    kg: KnowledgeGraph | None,
    spatial_map: PlaceGraph | None,
    epi_adapter: EpisodicMemoryAdapter | None,
    rel_adapter: RelationalMemoryAdapter | None,
    spat_adapter: SpatialAdapter | None,
) -> MemoryTokens | None:
    """Retrieve and concatenate memory tokens from enabled sources."""

    mems: list[MemoryTokens] = []
    dim = hidden.size(-1)

    if cfg.memory.episodic.enabled and store is not None and epi_adapter is not None:
        spec = TraceSpec(source="episodic", k=cfg.memory.episodic.k)
        mem = episodic_retrieve_and_pack(hidden, spec, store, epi_adapter.proj)
        logging.info(
            "episodic_retrieval_k=%d latency_ms=%.2f hit_rate=%.2f",
            spec.k or 0,
            mem.meta.get("latency_ms", 0.0),
            mem.meta.get("hit_rate", 0.0),
        )
        mems.append(mem)

    if cfg.memory.relational.enabled and kg is not None and rel_adapter is not None:
        proj = getattr(rel_adapter, "proj", None)
        if proj is None:
            base_dim = getattr(kg, "dim", dim)
            proj = nn.Linear(base_dim, dim)
            rel_adapter.proj = proj  # type: ignore[attr-defined]
        spec = TraceSpec(
            source="relational",
            k=cfg.memory.relational.k,
            params={"hops": cfg.memory.relational.hops},
        )
        mem = relational_retrieve_and_pack(hidden, spec, kg, proj)
        logging.info(
            "relational_retrieval_k=%d latency_ms=%.2f hit_rate=%.2f",
            spec.k or 0,
            mem.meta.get("latency_ms", 0.0),
            mem.meta.get("hit_rate", 0.0),
        )
        mems.append(mem)

    if cfg.memory.spatial.enabled and spatial_map is not None and spat_adapter is not None:
        proj = getattr(spat_adapter, "proj", None)
        if proj is None:
            proj = nn.Linear(4, dim)
            spat_adapter.proj = proj  # type: ignore[attr-defined]
        spec = TraceSpec(
            source="spatial",
            params={
                "radius": cfg.memory.spatial.radius,
                "max_nodes": cfg.memory.spatial.max_nodes,
                "max_edges": cfg.memory.spatial.max_edges,
            },
        )
        mem = spatial_retrieve_and_pack("origin", spec, spatial_map, proj)
        logging.info(
            "spatial_retrieval_radius=%d latency_ms=%.2f num_nodes=%d",
            cfg.memory.spatial.radius,
            mem.meta.get("latency_ms", 0.0),
            mem.meta.get("num_nodes", 0),
        )
        mems.append(mem)

    if not mems:
        return None
    tokens = torch.cat([m.tokens for m in mems], dim=1)
    mask = torch.cat([m.mask for m in mems], dim=1)
    meta = {"sources": [m.meta for m in mems]}
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)


def train(cfg: TrainConfig) -> None:
    """Execute the training or dry‑run."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    logging.info("Data format: %s", cfg.data_format)
    logging.info("Train files: %s", cfg.train_files)
    logging.info("Validation files: %s", cfg.val_files)
    logging.info("Fusion insert block index: %d", cfg.fusion_insert_block_index)
    logging.info("Replay enabled: %s", cfg.replay.enabled)
    logging.info("Replay ratio: %.2f", cfg.replay.ratio)

    model, tokenizer = _load_model_and_tokenizer(cfg)

    hidden = getattr(model.config, "hidden_size", cfg.episodic.hidden_size)

    store_cfg = {
        "hopfield": cfg.episodic.hopfield,
        "decay_rate": cfg.episodic_mem.decay_rate,
        "prune": {
            "min_salience": cfg.episodic_mem.prune_min_salience,
            "max_age": cfg.episodic_mem.prune_max_age,
        },
    }
    store = EpisodicStore(hidden, config=store_cfg)
    writer = AsyncStoreWriter(store)
    gate = WriteGate(tau=cfg.write_threshold)
    gate_attempts = 0
    gate_accepts = 0
    epi_interval = 0.1 if cfg.dry_run else cfg.episodic_mem.maintenance_interval
    store.start_background_tasks(epi_interval)

    epi_adapter: EpisodicMemoryAdapter | None = None
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
            hopfield=cfg.episodic.hopfield,
        )
        retrieval_dim = getattr(store, "dim", hidden)
        epi_adapter = EpisodicAdapter(epi_cfg)
        if not hasattr(epi_adapter, "proj"):
            epi_adapter.proj = nn.Linear(retrieval_dim, hidden)

    rel_adapter: RelationalMemoryAdapter | None = None
    if cfg.relational:
        rel_adapter = RelationalMemoryAdapter()

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

    fusion_cfg = MemoryFusionConfig(
        insert_block_index=cfg.fusion_insert_block_index,
        use_episodic=True,
        use_relational=cfg.relational,
        use_spatial=cfg.spatial.enabled,
    )

    class _SpatialProxy(nn.Module):
        def __init__(self, adapter: SpatialAdapter) -> None:
            super().__init__()
            self.adapter = adapter

        def forward(self, hidden_states: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            bsz, _, dim = hidden_states.shape
            plans = torch.zeros(
                bsz,
                1,
                dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            return self.adapter(hidden_states, plans)

    epi_proxy = epi_adapter
    rel_proxy = rel_adapter
    spat_proxy = _SpatialProxy(spat_adapter) if spat_adapter else None

    try:
        info = attach_adapters(
            model,
            fusion_cfg,
            episodic=epi_proxy,
            relational=rel_proxy,
            spatial=spat_proxy,
        )
        logging.info(
            "Adapter fusion attached at block %s/%s",
            info.get("target_block"),
            info.get("num_blocks"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Adapter fusion attach failed: %s", exc)

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

            if cfg.train_files:
                train_ds = load_jsonl_files(cfg.train_files)
                if cfg.val_files:
                    val_ds = load_jsonl_files(cfg.val_files)
                else:
                    train_ds, val_ds = split_train_val(cfg.train_files[0], None)
                logging.info("Train dataset size: %d", len(train_ds))
                logging.info("Validation dataset size: %d", len(val_ds))
            elif not cfg.dry_run:
                raise ValueError("train_files must be provided when data_format='jsonl'")
            else:
                logging.info("No train_files provided; skipping dataset load")
        elif not cfg.dry_run:
            train_ds = load_dataset(cfg.dataset_name, split="train")
            logging.info("Train dataset size: %d", len(train_ds))

        if cfg.memory.relational.enabled and kg is not None and train_ds is not None:
            ingested = 0
            for item in train_ds:
                text = item.get("text") or item.get("prompt") or ""
                ingested += ingest_text(
                    text,
                    kg,
                    fasttrack=cfg.memory.relational.schema_fasttrack,
                )
            logging.info("kg_ingested_tuples=%d", ingested)

        if cfg.replay.enabled and train_ds is not None:
            from scripts.replay_dataset import ReplayIterableDataset

            def replay_reader():
                while True:
                    kind, trace_id = scheduler.next_batch(1)[0]
                    yield {"prompt": f"replay {kind}", "answer": trace_id or ""}

            train_ds = ReplayIterableDataset(train_ds, replay_reader, cfg.replay.ratio)
            logging.info("Replay mixing active (ratio=%.2f)", cfg.replay.ratio)

        if cfg.dry_run:
            if peft_config is not None:
                model = get_peft_model(model, peft_config)
                count = count_trainable_parameters(model)
                logging.info("Trainable parameters: %d", count)
                if count <= 0:
                    raise RuntimeError(
                        "No trainable parameters found. Set `target_modules` to attach LoRA.",
                    )
            device = torch.device("cpu")
            try:
                device = next(model.parameters()).device  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - defensive
                pass
            dummy_hidden = torch.zeros(1, 1, hidden, device=device)
            mem = _gather_memory_tokens(
                dummy_hidden,
                cfg,
                store,
                kg,
                spatial_map,
                epi_adapter,
                rel_adapter,
                spat_adapter,
            )
            if mem is not None:
                dummy_ids = torch.ones(1, 1, dtype=torch.long, device=device)
                _ = model(dummy_ids, labels=dummy_ids, memory_tokens=mem)
            probs = np.array([1.0])
            queries = np.zeros((1, hidden), dtype=np.float32)
            keys = np.zeros((0, hidden), dtype=np.float32)
            decisions, _ = gate_batch(gate, probs, queries, keys)
            gate_attempts += len(decisions)
            gate_accepts += sum(d.allow for d in decisions)
            for dec, q in zip(decisions, queries):
                if dec.allow:
                    writer.enqueue(q, TraceValue(provenance="dry_run"))
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
        orig_compute_loss = trainer.compute_loss

        def compute_loss_with_memory(self, model, inputs, return_outputs=False):
            input_ids = inputs.get("input_ids")
            mem = None
            embed_fn = getattr(model, "get_input_embeddings", None)
            if callable(embed_fn) and input_ids is not None:
                hidden_emb = embed_fn()(input_ids)
                mem = _gather_memory_tokens(
                    hidden_emb,
                    cfg,
                    store,
                    kg,
                    spatial_map,
                    epi_adapter,
                    rel_adapter,
                    spat_adapter,
                )
            model._hippo_memory_tokens = mem  # type: ignore[attr-defined]
            return orig_compute_loss(model, inputs, return_outputs=return_outputs)

        trainer.compute_loss = compute_loss_with_memory.__get__(trainer, type(trainer))
        count = count_trainable_parameters(trainer.model)
        logging.info("Trainable parameters: %d", count)
        if count <= 0:
            raise RuntimeError(
                "No trainable parameters found. Set `target_modules` to attach LoRA."
            )

        try:
            trainer.train()
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
    finally:
        if worker is not None:
            worker.stop()
            worker.join(timeout=1)
        if "writer" in locals() and writer is not None:
            writer.stop()
            rate = gate_accepts / gate_attempts if gate_attempts else 0.0
            logging.info(
                "write_accept_rate=%.2f writes_enqueued=%d writes_committed=%d",
                rate,
                writer.stats["writes_enqueued"],
                writer.stats["writes_committed"],
            )


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
