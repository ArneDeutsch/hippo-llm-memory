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
from typing import Any, Iterable, List, Optional, Protocol

# Ensure repo root is on the path when executed as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from hippo_mem.adapters.episodic_adapter import EpisodicMemoryAdapter
from hippo_mem.adapters.lora import count_trainable_parameters, default_target_modules
from hippo_mem.adapters.patch import MemoryFusionConfig, attach_adapters
from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
from hippo_mem.common import MemoryTokens, ProvenanceLogger, TraceSpec
from hippo_mem.common.telemetry import gate_registry
from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic import episodic_retrieve_and_pack
from hippo_mem.episodic.adapter import AdapterConfig
from hippo_mem.episodic.async_writer import AsyncStoreWriter
from hippo_mem.episodic.gating import WriteGate, gate_batch
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore, TraceValue
from hippo_mem.relational.extract import extract_tuples
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.retrieval import relational_retrieve_and_pack
from hippo_mem.spatial.adapter import AdapterConfig as SpatialAdapterConfig
from hippo_mem.spatial.adapter import SpatialAdapter
from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.map import PlaceGraph
from hippo_mem.spatial.retrieval import spatial_retrieve_and_pack
from scripts.utils import log_memory_status

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
    gate: dict = field(default_factory=dict)


@dataclass
class SpatialSpec:
    enabled: bool = False
    radius: int = 1
    max_nodes: int = 0
    max_edges: int = 0
    gate: dict = field(default_factory=dict)


@dataclass
class RuntimeSpec:
    """Runtime toggles controlling side effects."""

    enable_writes: bool = True
    log_interval: int = 100


@dataclass
class Memory:
    episodic: EpisodicSpec = field(default_factory=EpisodicSpec)
    relational: RelationalSpec = field(default_factory=RelationalSpec)
    spatial: SpatialSpec = field(default_factory=SpatialSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)


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
    schema_fasttrack_ingest: bool = False

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
        fresh: float = 0.2  # kept: scheduler infers remainder; see tests/test_replay_scheduler.py

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


class MemorySource(Protocol):
    """Interface for lazily providing memory tokens."""

    def enabled(self, cfg: TrainConfig) -> bool:  # pragma: no cover - simple protocol
        ...

    def fetch(self, hidden: torch.Tensor) -> MemoryTokens:  # pragma: no cover - simple protocol
        ...


class EpisodicSource:
    def __init__(
        self,
        store: EpisodicStore | None,
        adapter: EpisodicMemoryAdapter | None,
        spec: EpisodicSpec,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.spec = spec

    def enabled(self, cfg: TrainConfig) -> bool:
        return self.spec.enabled and self.store is not None and self.adapter is not None

    def fetch(self, hidden: torch.Tensor) -> MemoryTokens:
        spec = TraceSpec(source="episodic", k=self.spec.k)
        mem = episodic_retrieve_and_pack(hidden, spec, self.store, self.adapter.proj)
        logging.info(
            "episodic_retrieval_k=%d latency_ms=%.2f hit_rate=%.2f",
            spec.k or 0,
            mem.meta.get("latency_ms", 0.0),
            mem.meta.get("hit_rate", 0.0),
        )
        return mem


class RelationalSource:
    def __init__(
        self,
        kg: KnowledgeGraph | None,
        adapter: RelationalMemoryAdapter | None,
        spec: RelationalSpec,
    ) -> None:
        self.kg = kg
        self.adapter = adapter
        self.spec = spec

    def enabled(self, cfg: TrainConfig) -> bool:
        return self.spec.enabled and self.kg is not None and self.adapter is not None

    def fetch(self, hidden: torch.Tensor) -> MemoryTokens:
        dim = hidden.size(-1)
        proj = getattr(self.adapter, "proj", None)
        if proj is None:
            base_dim = getattr(self.kg, "dim", dim)
            proj = nn.Linear(base_dim, dim)
            self.adapter.proj = proj  # type: ignore[attr-defined]
        spec = TraceSpec(
            source="relational",
            k=self.spec.k,
            params={"hops": self.spec.hops},
        )
        mem = relational_retrieve_and_pack(hidden, spec, self.kg, proj)
        logging.info(
            "relational_retrieval_k=%d latency_ms=%.2f hit_rate=%.2f",
            spec.k or 0,
            mem.meta.get("latency_ms", 0.0),
            mem.meta.get("hit_rate", 0.0),
        )
        return mem


class SpatialSource:
    def __init__(
        self,
        spatial_map: PlaceGraph | None,
        adapter: SpatialAdapter | None,
        spec: SpatialSpec,
    ) -> None:
        self.map = spatial_map
        self.adapter = adapter
        self.spec = spec

    def enabled(self, cfg: TrainConfig) -> bool:
        return self.spec.enabled and self.map is not None and self.adapter is not None

    def fetch(self, hidden: torch.Tensor) -> MemoryTokens:
        dim = hidden.size(-1)
        proj = getattr(self.adapter, "proj", None)
        if proj is None:
            proj = nn.Linear(4, dim)
            self.adapter.proj = proj  # type: ignore[attr-defined]
        spec = TraceSpec(
            source="spatial",
            params={
                "radius": self.spec.radius,
                "max_nodes": self.spec.max_nodes,
                "max_edges": self.spec.max_edges,
            },
        )
        mem = spatial_retrieve_and_pack("origin", spec, self.map, proj)
        logging.info(
            "spatial_retrieval_radius=%d latency_ms=%.2f num_nodes=%d num_edges=%d",
            self.spec.radius,
            mem.meta.get("latency_ms", 0.0),
            mem.meta.get("num_nodes", 0),
            mem.meta.get("num_edges", 0),
        )
        return mem


def _gather_memory_tokens(
    hidden: torch.Tensor, cfg: TrainConfig, sources: Iterable[MemorySource]
) -> MemoryTokens | None:
    """Retrieve and concatenate memory tokens from enabled sources."""

    mems = [src.fetch(hidden) for src in sources if src.enabled(cfg)]
    if not mems:
        return None
    tokens = torch.cat([m.tokens for m in mems], dim=1)
    mask = torch.cat([m.mask for m in mems], dim=1)
    meta = {"sources": [m.meta for m in mems]}
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)


@dataclass
class TrainerContext:
    """Holds model state and memory components for training."""

    model: nn.Module
    store: EpisodicStore | None = None
    writer: AsyncStoreWriter | None = None
    gate: WriteGate | None = None
    kg: KnowledgeGraph | None = None
    spatial_map: PlaceGraph | None = None
    scheduler: ReplayScheduler | None = None
    worker: ConsolidationWorker | None = None
    episodic_adapter: EpisodicMemoryAdapter | None = None
    relational_adapter: RelationalMemoryAdapter | None = None
    spatial_adapter: SpatialAdapter | None = None
    sources: list[MemorySource] = field(default_factory=list)
    hidden_size: int = 0
    gate_attempts: int = 0
    gate_accepts: int = 0
    step: int = 0
    log_interval: int = 100
    store_size: int = 0
    writer_queue_depth: int = 0
    adapters_attached: bool = False

    def start(self) -> None:
        if self.worker is not None:
            self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=1)
        if self.writer is not None:
            self.writer.stop()
            rate = self.gate_accepts / self.gate_attempts if self.gate_attempts else 0.0
            logging.info(
                "write_accept_rate=%.2f writes_enqueued=%d writes_committed=%d",
                rate,
                self.writer.stats["writes_enqueued"],
                self.writer.stats["writes_committed"],
            )
        if self.kg is not None:
            if hasattr(self.kg, "stop_background_tasks"):
                self.kg.stop_background_tasks()
            if hasattr(self.kg, "conn"):
                self.kg.conn.close()
        if self.spatial_map is not None and hasattr(self.spatial_map, "stop_background_tasks"):
            self.spatial_map.stop_background_tasks()

    def runtime_stats(self) -> dict[str, int]:
        """Return latest runtime counters for external monitoring."""

        return {
            "step": self.step,
            "gate_accepts": self.gate_accepts,
            "gate_attempts": self.gate_attempts,
            "store_size": self.store_size,
            "writer_queue_depth": self.writer_queue_depth,
        }


def ingest_training_text(records: Iterable[dict[str, Any]], kg: KnowledgeGraph) -> None:
    """Extract tuples from training ``records`` and insert into ``kg``.

    Parameters
    ----------
    records : Iterable[dict[str, Any]]
        Sequence of training examples containing a ``"text"`` field.
    kg : KnowledgeGraph
        Graph that receives ingested tuples.
    """

    for rec in records:
        text = rec.get("text") or ""
        for tup in extract_tuples(text):
            kg.ingest(tup)


def build_spatial_gate(
    gate_cfg: Optional[dict], logger: ProvenanceLogger | None = None
) -> SpatialGate | None:
    """Return a :class:`SpatialGate` from ``gate_cfg`` if enabled."""

    if gate_cfg and gate_cfg.get("enabled"):
        return SpatialGate(
            block_threshold=gate_cfg.get("block_threshold", 1.0),
            repeat_N=gate_cfg.get("repeat_N", 3),
            recent_window=gate_cfg.get("recent_window", 20),
            max_degree=gate_cfg.get("max_degree", 64),
            logger=logger,
        )
    return None


def process_coordinate(
    prev: str | None,
    ctx: str,
    gate: SpatialGate | None,
    graph: PlaceGraph,
    stats,
) -> str:
    """Apply gate decision for ``ctx`` and insert/aggregate as needed."""

    if gate is None:
        graph.observe(ctx)
    else:
        stats.attempts += 1
        action, _reason = gate.decide(prev, ctx, graph)
        if action == "insert":
            stats.inserted += 1
            graph.observe(ctx)
        elif action == "aggregate" and prev is not None:
            stats.aggregated += 1
            graph.aggregate_duplicate(prev, ctx)
        elif action == "route_to_episodic":
            stats.blocked_new_edges += 1
    return ctx


def ingest_spatial_traces(
    records: Iterable[dict[str, Any]],
    graph: PlaceGraph,
    gate: SpatialGate | None = None,
) -> None:
    """Insert trajectories from ``records`` into ``graph``.

    Parameters
    ----------
    records : Iterable[dict[str, Any]]
        Training examples optionally containing a ``"trajectory"`` field.
    graph : PlaceGraph
        Spatial map receiving the observed coordinates.
    gate : SpatialGate | None, optional
        Optional gating module controlling insert/aggregate decisions.
    """

    stats = gate_registry.get("spatial")

    count = 0
    steps = 0
    for rec in records:
        traj = rec.get("trajectory")
        if not traj:
            continue
        count += 1
        steps += len(traj)
        prev = None
        for coord in traj:
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                ctx = f"{coord[0]},{coord[1]}"
            else:  # pragma: no cover - defensive
                ctx = str(coord)
            prev = process_coordinate(prev, ctx, gate, graph, stats)
    if count:
        logging.info(
            "spatial_ingest_traces=%d avg_len=%.2f",
            count,
            steps / count if count else 0.0,
        )


class _IngestCallback(TrainerCallback):
    """Hydra/Transformers callback inserting KG tuples and spatial traces."""

    def __init__(
        self,
        cfg: TrainConfig,
        records: Iterable[dict[str, Any]] | None,
        kg: KnowledgeGraph,
        graph: PlaceGraph,
        provenance: ProvenanceLogger | None,
    ) -> None:
        self.cfg = cfg
        self.records = list(records) if records is not None else []
        self.kg = kg
        self.graph = graph
        self.provenance = provenance
        self._kg_nodes = kg.graph.number_of_nodes()
        self._kg_edges = kg.graph.number_of_edges()
        self._pg_nodes = len(graph.graph)
        self._pg_edges = sum(len(v) for v in graph.graph.values())

    def on_epoch_end(self, args, state, control, **kwargs):  # pragma: no cover - Trainer API
        epoch = int(state.epoch or 0)
        if self.cfg.schema_fasttrack_ingest and self.records:
            ingest_training_text(self.records, self.kg)
            nodes = self.kg.graph.number_of_nodes()
            edges = self.kg.graph.number_of_edges()
            logging.info(
                "kg_ingest_epoch=%d nodes=%d(+%d) edges=%d(+%d)",
                epoch,
                nodes,
                nodes - self._kg_nodes,
                edges,
                edges - self._kg_edges,
            )
            self._kg_nodes, self._kg_edges = nodes, edges
        if self.cfg.spatial.enabled and self.records:
            gate_cfg = getattr(self.cfg.memory.spatial, "gate", None)
            gate = build_spatial_gate(gate_cfg, self.provenance)
            ingest_spatial_traces(self.records, self.graph, gate)
            nodes = len(self.graph.graph)
            edges = sum(len(v) for v in self.graph.graph.values())
            logging.info(
                "spatial_ingest_epoch=%d nodes=%d(+%d) edges=%d(+%d)",
                epoch,
                nodes,
                nodes - self._pg_nodes,
                edges,
                edges - self._pg_edges,
            )
            self._pg_nodes, self._pg_edges = nodes, edges


def prepare_datasets(cfg: TrainConfig):
    """Load training and validation datasets based on ``cfg``."""

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
    return train_ds, val_ds


def setup_store(
    context: TrainerContext, cfg: TrainConfig, provenance: ProvenanceLogger | None = None
) -> None:
    """Configure episodic store, writer and gate."""
    model = context.model
    hidden = getattr(model.config, "hidden_size", cfg.episodic.hidden_size)
    context.hidden_size = hidden
    store_cfg = {
        "hopfield": cfg.episodic.hopfield,
        "decay_rate": cfg.episodic_mem.decay_rate,
        "prune": {
            "min_salience": cfg.episodic_mem.prune_min_salience,
            "max_age": cfg.episodic_mem.prune_max_age,
        },
    }
    context.store = EpisodicStore(hidden, config=store_cfg)
    context.writer = AsyncStoreWriter(context.store)
    context.gate = WriteGate(tau=cfg.write_threshold, logger=provenance)
    epi_interval = 0.1 if cfg.dry_run else cfg.episodic_mem.maintenance_interval
    context.store.start_background_tasks(epi_interval)


def setup_adapters(context: TrainerContext, cfg: TrainConfig) -> None:
    """Initialise memory adapters."""
    if not (cfg.episodic.enabled or cfg.relational or cfg.spatial.enabled):
        return
    hidden = context.hidden_size
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
        retrieval_dim = getattr(context.store, "dim", hidden)
        context.episodic_adapter = EpisodicAdapter(epi_cfg)
        if not hasattr(context.episodic_adapter, "proj"):
            context.episodic_adapter.proj = nn.Linear(retrieval_dim, hidden)
    if cfg.relational:
        context.relational_adapter = RelationalMemoryAdapter()
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
        context.spatial_adapter = SpatialAdapter(spat_cfg)


def attach_fusion(context: TrainerContext, cfg: TrainConfig) -> None:
    """Attach memory fusion patch and any configured adapters.

    The patch is installed even when no adapters are present so tests and
    smoke checks can verify fusion wiring independently of adapter toggles.
    """

    model = context.model
    fusion_cfg = MemoryFusionConfig(
        insert_block_index=cfg.fusion_insert_block_index,
        # Always enable episodic branch to mirror previous behaviour where the
        # fusion patch was attached regardless of adapter configuration.
        use_episodic=True,
        use_relational=context.relational_adapter is not None,
        use_spatial=context.spatial_adapter is not None,
    )

    class _RelationalProxy(nn.Module):
        """Temporary stand-in until full KG reader is integrated."""

        def __init__(self, adapter: RelationalMemoryAdapter) -> None:
            super().__init__()
            self.adapter = adapter

        def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return self.adapter(hidden_states, **kwargs)

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

    epi_proxy = context.episodic_adapter
    rel_proxy = _RelationalProxy(context.relational_adapter) if context.relational_adapter else None
    spat_proxy = _SpatialProxy(context.spatial_adapter) if context.spatial_adapter else None
    try:
        info = attach_adapters(
            model,
            fusion_cfg,
            episodic=epi_proxy,
            relational=rel_proxy,
            spatial=spat_proxy,
        )
        context.adapters_attached = True
        logging.info(
            "Adapter fusion attached at block %s/%s",
            info.get("target_block"),
            info.get("num_blocks"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        context.adapters_attached = False
        logging.warning("Adapter fusion attach failed: %s", exc)


def setup_sources(context: TrainerContext, cfg: TrainConfig) -> None:
    """Initialise knowledge graph, spatial map and memory sources."""
    sources: list[MemorySource] = []
    kg_cfg = {
        "prune": {
            "min_conf": cfg.relational_mem.prune_min_conf,
            "max_age": cfg.relational_mem.prune_max_age,
        }
    }
    gate_cfg = getattr(cfg.memory.relational, "gate", {})
    kg_gate = None
    if gate_cfg.get("enabled"):
        kg_gate = RelationalGate(
            threshold=gate_cfg.get("threshold", 0.6),
            w_conf=gate_cfg.get("w_conf", 0.6),
            w_nov=gate_cfg.get("w_nov", 0.5),
            w_deg=gate_cfg.get("w_deg", 0.4),
            w_rec=gate_cfg.get("w_rec", 0.2),
            max_degree=gate_cfg.get("max_degree", 64),
        )
        context.kg = KnowledgeGraph(config=kg_cfg, gate=kg_gate)
    else:
        context.kg = KnowledgeGraph(config=kg_cfg)
    kg_interval = 0.1 if cfg.dry_run else cfg.relational_mem.maintenance_interval
    context.kg.start_background_tasks(kg_interval)
    if cfg.memory.relational.enabled:
        sources.append(
            RelationalSource(context.kg, context.relational_adapter, cfg.memory.relational)
        )
    spat_map_cfg = {
        "decay_rate": cfg.spatial_mem.decay_rate,
        "prune": {"max_age": cfg.spatial_mem.prune_max_age},
    }
    context.spatial_map = PlaceGraph(path_integration=cfg.spatial.enabled, config=spat_map_cfg)
    spat_interval = 0.1 if cfg.dry_run else cfg.spatial_mem.maintenance_interval
    context.spatial_map.start_background_tasks(spat_interval)
    if cfg.memory.spatial.enabled:
        sources.append(
            SpatialSource(context.spatial_map, context.spatial_adapter, cfg.memory.spatial)
        )
    if cfg.memory.episodic.enabled:
        sources.append(EpisodicSource(context.store, context.episodic_adapter, cfg.memory.episodic))
    context.sources = sources


def setup_replay(context: TrainerContext, cfg: TrainConfig, model: nn.Module) -> None:
    """Configure replay scheduler and worker."""
    if not cfg.replay.enabled:
        context.scheduler = None
        context.worker = None
        return
    hidden = context.hidden_size
    context.scheduler = ReplayScheduler(context.store, context.kg, batch_mix=cfg.batch_mix)
    for i in range(10):
        context.scheduler.add_trace(str(i), np.zeros(hidden, dtype=np.float32), score=float(i))
    context.worker = ConsolidationWorker(
        context.scheduler,
        model,
        episodic_adapter=context.episodic_adapter,
        relational_adapter=context.relational_adapter,
        spatial_adapter=context.spatial_adapter,
    )


@dataclass
class MemoryInitializer:
    """Facade around memory setup helpers."""

    context: TrainerContext
    cfg: TrainConfig
    provenance: ProvenanceLogger | None = None

    def initialize(self) -> None:
        setup_store(self.context, self.cfg, self.provenance)
        setup_adapters(self.context, self.cfg)
        attach_fusion(self.context, self.cfg)
        setup_sources(self.context, self.cfg)
        setup_replay(self.context, self.cfg, self.context.model)


def init_memory(
    context: TrainerContext,
    cfg: TrainConfig,
    provenance: ProvenanceLogger | None = None,
) -> None:
    """Configure memory components via :class:`MemoryInitializer`."""
    MemoryInitializer(context, cfg, provenance).initialize()


def _train(
    cfg: TrainConfig,
    context: TrainerContext,
    tokenizer,
    train_ds,
    val_ds,
    provenance: ProvenanceLogger | None,
) -> None:
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

    model = context.model
    hidden = context.hidden_size
    sources = context.sources

    if context.adapters_attached:

        def _block_retrieval_cb(hs: torch.Tensor) -> MemoryTokens | None:
            setattr(model, "_hippo_last_hidden", hs.detach())
            return _gather_memory_tokens(hs, cfg, sources)

        setattr(model, "_hippo_retrieval_cb", _block_retrieval_cb)  # type: ignore[attr-defined]

    context.start()
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

        if cfg.replay.enabled and train_ds is not None and context.scheduler is not None:
            from scripts.replay_dataset import ReplayIterableDataset

            def replay_reader():
                while True:
                    kind, trace_id = context.scheduler.next_batch(1)[0]
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
            mem = _gather_memory_tokens(dummy_hidden, cfg, sources)
            dummy_ids = torch.ones(1, 1, dtype=torch.long, device=device)
            if callable(model):
                _ = model(dummy_ids, labels=dummy_ids, memory_tokens=mem)
            if cfg.memory.runtime.enable_writes:
                probs = np.array([1.0])
                queries = np.zeros((1, hidden), dtype=np.float32)
                keys = np.zeros((0, hidden), dtype=np.float32)
                decisions, _ = gate_batch(context.gate, probs, queries, keys)
                context.gate_attempts += len(decisions)
                context.gate_accepts += sum(d.allow for d in decisions)
                for dec, q in zip(decisions, queries):
                    if dec.allow and context.writer is not None:
                        context.writer.enqueue(q, TraceValue(provenance="dry_run"))
            for _ in range(3):
                log_memory_status(
                    context.store,
                    context.kg,
                    context.spatial_map,
                    context.scheduler,
                    context.worker,
                )
                time.sleep(0.1)
            if train_ds is not None:
                if cfg.schema_fasttrack_ingest:
                    ingest_training_text(train_ds, context.kg)
                if cfg.spatial.enabled:
                    gate_cfg = getattr(cfg.memory.spatial, "gate", None)
                    gate = build_spatial_gate(gate_cfg, provenance)
                    ingest_spatial_traces(train_ds, context.spatial_map, gate)
            return

        training_args = SFTConfig(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            logging_steps=1,
            bf16=torch.cuda.is_available(),
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
        if hasattr(trainer, "add_callback"):
            trainer.add_callback(
                _IngestCallback(cfg, train_ds, context.kg, context.spatial_map, provenance)
            )
        orig_compute_loss = trainer.compute_loss

        def compute_loss_with_memory(self, model, inputs, return_outputs=False):
            mem_cb = getattr(model, "_hippo_retrieval_cb", None)
            inputs = dict(inputs)
            if not callable(mem_cb):
                input_ids = inputs.get("input_ids")
                mem = None
                embed_fn = getattr(model, "get_input_embeddings", None)
                if callable(embed_fn) and input_ids is not None:
                    hidden_emb = embed_fn()(input_ids)
                    mem = _gather_memory_tokens(hidden_emb, cfg, sources)
                if mem is not None:
                    inputs["memory_tokens"] = mem
            else:
                setattr(model, "_hippo_memory_tokens", None)
            loss, outputs = orig_compute_loss(model, inputs, return_outputs=True)
            gate, store, writer = context.gate, context.store, context.writer
            context.step += 1
            hidden = getattr(model, "_hippo_last_hidden", None)
            labels = inputs.get("labels")
            if (
                cfg.memory.runtime.enable_writes
                and gate
                and store
                and writer is not None
                and hidden is not None
                and labels is not None
            ):
                with torch.no_grad():
                    last_logits = outputs.logits[:, -1, :]
                    last_labels = labels[:, -1]
                    log_probs = F.log_softmax(last_logits, dim=-1)
                    probs = (
                        torch.exp(log_probs[torch.arange(last_labels.size(0)), last_labels])
                        .cpu()
                        .numpy()
                    )
                    queries = hidden[:, -1, :].detach().cpu().numpy()
                    keys = store.keys()
                decisions, _ = gate_batch(gate, probs, queries, keys)
                context.gate_attempts += len(decisions)
                context.gate_accepts += sum(d.allow for d in decisions)
                for dec, q in zip(decisions, queries):
                    if dec.allow:
                        writer.enqueue(q, TraceValue(provenance="train"))
                context.store_size = keys.shape[0]
                context.writer_queue_depth = writer.queue.qsize()
                if context.step % context.log_interval == 0:
                    logging.info(
                        "step=%d gate_accepts=%d gate_attempts=%d store_size=%d queue_depth=%d",
                        context.step,
                        context.gate_accepts,
                        context.gate_attempts,
                        context.store_size,
                        context.writer_queue_depth,
                    )
            if return_outputs:
                return loss, outputs
            return loss

        trainer.compute_loss = compute_loss_with_memory.__get__(trainer, type(trainer))
        count = count_trainable_parameters(trainer.model)
        logging.info("Trainable parameters: %d", count)
        if count <= 0:
            raise RuntimeError(
                "No trainable parameters found. Set `target_modules` to attach LoRA."
            )

        trainer.train()

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    finally:
        context.stop()


def train(cfg: TrainConfig) -> None:
    """Wrapper that prepares context and datasets then runs training."""

    model, tokenizer = _load_model_and_tokenizer(cfg)
    context = TrainerContext(model=model, log_interval=cfg.memory.runtime.log_interval)
    provenance = ProvenanceLogger(os.getcwd())
    MemoryInitializer(context, cfg, provenance).initialize()
    train_ds, val_ds = prepare_datasets(cfg)
    _train(cfg, context, tokenizer, train_ds, val_ds, provenance)


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
