from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from omegaconf import DictConfig
from torch import Tensor, nn

from hippo_eval.bench import _flatten_ablate, _init_modules
from hippo_mem.common import TraceSpec
from hippo_mem.common.gates import GateCounters
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.retrieval import relational_retrieve_and_pack
from hippo_mem.relational.tuples import extract_tuples

from .base import EvalAdapter, RetrieveResult


@dataclass
class RelationalEvalAdapter(EvalAdapter):
    """Evaluation adapter wrapping relational memory operations."""

    def present(self) -> str:
        return "relational"

    def build(self, cfg: DictConfig) -> Dict[str, object]:
        mem_cfg = getattr(cfg, "memory", None)
        ablate = _flatten_ablate(getattr(cfg, "ablate", {}))
        modules = _init_modules(mem_cfg, ablate)
        rel = modules.get("relational", {})
        if "gate" not in rel:
            rel["gate"] = RelationalGate()
        return rel

    def retrieve(
        self,
        cfg: DictConfig,
        modules: Dict[str, object],
        item: object,
        *,
        context_key: str,
        hidden: Tensor,
    ) -> RetrieveResult:
        kg = modules["kg"]
        adapter = modules["adapter"]
        hidden_dim = hidden.size(-1)
        in_dim = getattr(kg, "dim", 0) or hidden_dim
        proj = getattr(adapter, "proj", nn.Linear(in_dim, hidden_dim))
        adapter.proj = proj  # type: ignore[attr-defined]
        spec = TraceSpec(source="relational", k=1)
        mem = relational_retrieve_and_pack(hidden, spec, kg, proj, context_keys=[context_key])
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
        gate: RelationalGate | None = modules.get("gate")
        kg = modules["kg"]
        texts: list[str] = []
        text = getattr(item, "fact", None)
        facts = getattr(item, "facts", None)
        if text:
            texts = [text]
        elif facts:
            texts = [f.get("text", "") for f in facts]
        for txt in texts:
            for tup in extract_tuples(txt, threshold=0.2):
                if gate is not None:
                    decision = gate.decide(tup, kg)
                    gc.attempts += 1
                    if decision.action == "insert":
                        if not dry_run:
                            kg.schema_index.fast_track(tup, kg)
                        gc.accepted += 1
                    else:
                        gc.skipped += 1
                else:
                    if not dry_run:
                        kg.schema_index.fast_track(tup, kg)

    def store_size(self, modules: Dict[str, object]) -> Tuple[int, Dict[str, int]]:
        kg = modules["kg"]
        size = int(kg.graph.number_of_edges())
        status = kg.log_status()
        diag = {
            "nodes_added": int(status.get("nodes_added", kg.graph.number_of_nodes())),
            "edges_added": int(status.get("edges_added", kg.graph.number_of_edges())),
            "coref_merges": int(status.get("coref_merges", 0)),
        }
        return size, diag


__all__ = ["RelationalEvalAdapter"]
