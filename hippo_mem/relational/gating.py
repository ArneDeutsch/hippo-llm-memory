"""Lightweight salience gate for relational tuples.

Summary
-------
Down-weights duplicate or low-novelty tuples before they reach the
:class:`~hippo_mem.relational.schema.SchemaIndex`.  The gate combines
confidence, novelty, node degree and recency into a single score and
admits tuples whose score crosses a threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Tuple

from .tuples import TupleType

if TYPE_CHECKING:  # pragma: no cover - import cycle hints
    from .kg import KnowledgeGraph


@dataclass
class RelationalGate:
    """Score tuples and block low-salience inserts."""

    threshold: float = 0.6
    w_conf: float = 0.6
    w_nov: float = 0.5
    w_deg: float = 0.4
    w_rec: float = 0.2
    max_degree: int = 64
    recency_window: float = 60.0
    _last_seen: Dict[str, float] = field(default_factory=dict)

    def decide(self, tup: TupleType, kg: "KnowledgeGraph") -> Tuple[str, str]:
        """Return ``(action, reason)`` for ``tup``."""

        head, rel, tail, *_rest, conf, _prov = tup

        novelty = 1.0
        if kg.graph.has_edge(head, tail):
            data = kg.graph.get_edge_data(head, tail) or {}
            for edge in data.values():
                if edge.get("relation") == rel:
                    novelty = 0.0
                    break

        deg_pen = 0.0
        for node in (head, tail):
            deg = kg.graph.degree(node) if kg.graph.has_node(node) else 0
            if deg > self.max_degree:
                deg_pen = max(deg_pen, (deg - self.max_degree) / self.max_degree)

        now = time.time()
        rec_h = now - self._last_seen.get(head, 0.0)
        rec_t = now - self._last_seen.get(tail, 0.0)
        rec_bonus = 1.0 if rec_h > self.recency_window and rec_t > self.recency_window else 0.0

        score = (
            self.w_conf * conf
            + self.w_nov * novelty
            - self.w_deg * deg_pen
            + self.w_rec * rec_bonus
        )

        self._last_seen[head] = now
        self._last_seen[tail] = now

        if novelty == 0.0:
            return "aggregate", "duplicate_edge"
        if score >= self.threshold:
            return "insert", f"score={score:.2f}>=thr"
        return "route_to_episodic", f"score={score:.2f}<thr"


__all__ = ["RelationalGate"]
