from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from omegaconf import DictConfig
from torch import Tensor, nn

from hippo_eval.bench import _flatten_ablate, _init_modules
from hippo_mem.common import TraceSpec
from hippo_mem.common.gates import GateCounters
from hippo_mem.episodic.gating import DGKey, WriteGate, k_wta
from hippo_mem.episodic.retrieval import episodic_retrieve_and_pack
from hippo_mem.episodic.types import TraceValue
from hippo_mem.retrieval.embed import embed_text

from .base import EvalAdapter, RetrieveResult


@dataclass
class EpisodicEvalAdapter(EvalAdapter):
    """Evaluation adapter wrapping episodic memory operations."""

    def present(self) -> str:
        return "episodic"

    # Build episodic modules via existing helper
    def build(self, cfg: DictConfig) -> Dict[str, object]:
        mem_cfg = getattr(cfg, "memory", None)
        ablate = _flatten_ablate(getattr(cfg, "ablate", {}))
        modules = _init_modules(mem_cfg, ablate)
        epi = modules.get("episodic", {})
        if "gate" not in epi:
            epi["gate"] = WriteGate()
        return epi

    def retrieve(
        self,
        cfg: DictConfig,
        modules: Dict[str, object],
        item: object,
        *,
        context_key: str,
        hidden: Tensor,
    ) -> RetrieveResult:
        store = modules["store"]
        adapter = modules["adapter"]
        proj = getattr(adapter, "proj", nn.Linear(store.dim, hidden.size(-1)))
        adapter.proj = proj  # type: ignore[attr-defined]
        spec = TraceSpec(source="episodic", k=1)
        mem = episodic_retrieve_and_pack(hidden, spec, store, proj, context_keys=[context_key])
        hit = bool(mem.meta.get("hit_rate", 0.0) > 0)
        latency = float(mem.meta.get("latency_ms", 0.0))
        topk = mem.meta.get("trace_ids", []) or []
        return RetrieveResult(mem=mem, hit=hit, latency_ms=latency, topk_keys=topk)

    def teach(
        self,
        cfg: DictConfig,
        modules: Dict[str, object],
        item: object,
        *,
        dry_run: bool,
        gc: GateCounters,
        suite: str,
    ) -> None:
        gate: WriteGate | None = modules.get("gate")
        store = modules["store"]
        if gate is None:
            return
        text = getattr(item, "fact", None) or getattr(item, "prompt", "")
        if not text:
            return
        k = getattr(store, "k_wta", 0)
        dense, sparse = episodic_key_from_text(text, store.dim, k)
        decision = gate.decide(0.5, dense, store.keys(), provenance="teach")
        gc.attempts += 1
        if decision.action == "insert":
            if not dry_run:
                value = TraceValue(
                    provenance="teach", context_key=getattr(item, "context_key", None), suite=suite
                )
                key = sparse if sparse is not None else dense
                store.write(key, value, context_key=getattr(item, "context_key", None))
            gc.accepted += 1
        else:
            gc.skipped += 1

    def store_size(self, modules: Dict[str, object]) -> Tuple[int, Dict[str, int]]:
        store = modules["store"]
        try:
            cur = store.persistence.db.conn.cursor()
            cur.execute("SELECT value FROM traces")
            rows = cur.fetchall()
            size = sum(1 for (val,) in rows if json.loads(val or "{}").get("provenance") != "dummy")
        except Exception:
            size = 0
        return size, {}


def episodic_key_from_text(text: str, dim: int, k: int) -> tuple[np.ndarray, DGKey]:
    """Return dense and sparse keys derived from ``text``."""

    vec = np.array(embed_text(text, dim=dim), dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    sparse = (
        k_wta(vec, k)
        if k > 0
        else DGKey(np.array([], dtype="int64"), np.array([], dtype="float32"), dim)
    )
    return vec, sparse


__all__ = ["EpisodicEvalAdapter", "episodic_key_from_text"]
