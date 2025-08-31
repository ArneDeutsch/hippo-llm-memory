"""Gating heuristics for spatial observations.

Summary
-------
Blocks repeated self-observations and rapid flapping between the same
pair of places.  Each observation is scored based on repetition,
recent edge use, and node degree before being written to the
:class:`~hippo_mem.spatial.map.PlaceGraph`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from hippo_mem.common import GateDecision, ProvenanceLogger, log_gate
from hippo_mem.common.telemetry import gate_registry

from .map import PlaceGraph


@dataclass
class SpatialGate:
    """Reject degenerate spatial observations."""

    block_threshold: float = 1.0
    repeat_N: int = 3
    recent_window: int = 20
    max_degree: int = 64
    logger: ProvenanceLogger | None = None
    _ctx_hist: Deque[str] = field(init=False)
    _edge_hist: Deque[tuple[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.block_threshold <= 1.0:
            raise ValueError("block_threshold must be in [0, 1]")
        if self.repeat_N < 1:
            raise ValueError("repeat_N must be >= 1")
        if self.recent_window < 1:
            raise ValueError("recent_window must be >= 1")
        if self.max_degree < 1:
            raise ValueError("max_degree must be >= 1")
        self._ctx_hist = deque(maxlen=self.repeat_N)
        self._edge_hist = deque(maxlen=self.recent_window)

    def decide(self, prev_ctx: Optional[str], context: str, graph: PlaceGraph) -> GateDecision:
        """Return a :class:`GateDecision` for a context transition."""
        stats = gate_registry.get("spatial")
        stats.attempts += 1

        repeat_pen = 0.0
        if len(self._ctx_hist) >= self.repeat_N - 1 and all(
            c == context for c in list(self._ctx_hist)[-(self.repeat_N - 1) :]
        ):
            repeat_pen = 1.0

        edge_pen = 0.0
        if prev_ctx is not None and (prev_ctx, context) in self._edge_hist:
            edge_pen = 1.0

        deg_pen = 0.0
        node_id = graph._context_to_id.get(context)  # noqa: SLF001 - read-only
        if node_id is not None:
            deg = len(graph.graph.get(node_id, {}))
            if deg > self.max_degree:
                deg_pen = (deg - self.max_degree) / self.max_degree

        score = repeat_pen + edge_pen + deg_pen

        self._ctx_hist.append(context)
        if score >= self.block_threshold:
            decision = GateDecision("route_to_episodic", f"score={score:.2f}>=thr", score)
            stats.blocked_new_edges += 1
        else:
            action = "insert"
            reason = "new_edge"
            if prev_ctx is not None:
                a_id = graph._context_to_id.get(prev_ctx)
                b_id = graph._context_to_id.get(context)
                if a_id is not None and b_id is not None and graph.graph.get(a_id, {}).get(b_id):
                    action = "aggregate"
                    reason = "duplicate_edge"
                self._edge_hist.append((prev_ctx, context))
            decision = GateDecision(action, reason, score)
            if action == "insert":
                stats.inserted += 1
            elif action == "aggregate":
                stats.aggregated += 1
        log_gate(
            self.logger,
            "spatial",
            decision,
            {"prev": prev_ctx, "ctx": context, "deg_pen": deg_pen, "score": score},
        )
        return decision


__all__ = ["SpatialGate"]
