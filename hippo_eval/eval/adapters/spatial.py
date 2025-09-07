from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

from omegaconf import DictConfig
from torch import Tensor, nn

from hippo_eval.bench import _flatten_ablate, _init_modules
from hippo_mem.common import TraceSpec
from hippo_mem.common.gates import GateCounters
from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.retrieval import spatial_retrieve_and_pack

from .base import EvalAdapter, RetrieveResult


@dataclass
class SpatialEvalAdapter(EvalAdapter):
    """Evaluation adapter wrapping spatial memory operations."""

    def present(self) -> str:
        return "spatial"

    def build(self, cfg: DictConfig) -> Dict[str, object]:
        mem_cfg = getattr(cfg, "memory", None)
        ablate = _flatten_ablate(getattr(cfg, "ablate", {}))
        modules = _init_modules(mem_cfg, ablate)
        spat = modules.get("spatial", {})
        if "gate" not in spat:
            spat["gate"] = SpatialGate()
        return spat

    def retrieve(
        self,
        cfg: DictConfig,
        modules: Dict[str, object],
        item: object,
        *,
        context_key: str,
        hidden: Tensor,
    ) -> RetrieveResult:
        graph = modules["map"]
        adapter = modules["adapter"]
        proj = getattr(adapter, "proj", nn.Linear(4, hidden.size(-1)))
        adapter.proj = proj  # type: ignore[attr-defined]
        spec = TraceSpec(source="spatial")
        mem = spatial_retrieve_and_pack("origin", spec, graph, proj, context_keys=[context_key])
        hit = bool(mem.meta.get("hit_rate", 0.0) > 0)
        latency = float(mem.meta.get("latency_ms", 0.0))
        topk: list[str] = mem.meta.get("trace_ids", []) or []
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
        gate: SpatialGate | None = modules.get("gate")
        if gate is None:
            return
        graph = modules["map"]
        m = re.search(r"Start \((\d+),\s*(\d+)\)", getattr(item, "prompt", ""))
        if not m:
            return
        x, y = map(int, m.groups())
        prev: str | None = None
        ctx = (
            f"{getattr(item, 'context_key', None)}:{x},{y}"
            if getattr(item, "context_key", None)
            else f"{x},{y}"
        )
        decision = gate.decide(prev, ctx, graph)
        gc.attempts += 1
        if decision.action == "insert":
            if not dry_run:
                graph.observe(ctx)
            gc.accepted += 1
        else:
            gc.skipped += 1
        prev = ctx
        moves = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        for step in getattr(item, "answer", ""):
            dx, dy = moves.get(step, (0, 0))
            x += dx
            y += dy
            ctx = (
                f"{getattr(item, 'context_key', None)}:{x},{y}"
                if getattr(item, "context_key", None)
                else f"{x},{y}"
            )
            decision = gate.decide(prev, ctx, graph)
            gc.attempts += 1
            if decision.action == "insert":
                if not dry_run:
                    graph.observe(ctx)
                gc.accepted += 1
            elif decision.action == "aggregate":
                if not dry_run and prev is not None:
                    graph.aggregate_duplicate(prev, ctx)
                gc.skipped += 1
            else:
                gc.skipped += 1
            prev = ctx

    def store_size(self, modules: Dict[str, object]) -> Tuple[int, Dict[str, int]]:
        g = modules["map"]
        nodes = len(getattr(g, "_context_to_id", {}))
        edges = sum(len(nbrs) for nbrs in getattr(g, "graph", {}).values())
        diag = {"writes": int(g.log_status().get("writes", 0))}
        return int(nodes + edges), diag


__all__ = ["SpatialEvalAdapter"]
