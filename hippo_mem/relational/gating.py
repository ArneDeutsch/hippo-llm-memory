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

from hippo_mem.common import ProvenanceLogger

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
    logger: ProvenanceLogger | None = None
    _last_seen: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        for name in ("w_conf", "w_nov", "w_deg", "w_rec"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_degree < 1:
            raise ValueError("max_degree must be >= 1")
        if self.recency_window <= 0:
            raise ValueError("recency_window must be > 0")

    def _novelty(self, head: str, rel: str, tail: str, kg: "KnowledgeGraph") -> float:
        """Return 1.0 for novel edges and 0.0 for duplicates."""

        if kg.graph.has_edge(head, tail):
            data = kg.graph.get_edge_data(head, tail) or {}
            for edge in data.values():
                if edge.get("relation") == rel:
                    return 0.0
        return 1.0

    def _degree_penalty(self, head: str, tail: str, kg: "KnowledgeGraph") -> float:
        """Penalty for exceeding ``max_degree`` on either node."""

        deg_pen = 0.0
        for node in (head, tail):
            deg = kg.graph.degree(node) if kg.graph.has_node(node) else 0
            if deg > self.max_degree:
                deg_pen = max(deg_pen, (deg - self.max_degree) / self.max_degree)
        return deg_pen

    def _recency_bonus(self, head: str, tail: str) -> float:
        """Bonus if both nodes fall outside the recency window."""

        now = time.time()
        rec_h = now - self._last_seen.get(head, 0.0)
        rec_t = now - self._last_seen.get(tail, 0.0)
        self._last_seen[head] = now
        self._last_seen[tail] = now
        return 1.0 if rec_h > self.recency_window and rec_t > self.recency_window else 0.0

    def decide(self, tup: TupleType, kg: "KnowledgeGraph") -> Tuple[str, str]:
        """Return ``(action, reason)`` for ``tup``."""

        head, rel, tail, *_rest, conf, _prov = tup

        novelty = self._novelty(head, rel, tail, kg)
        deg_pen = self._degree_penalty(head, tail, kg)
        rec_bonus = self._recency_bonus(head, tail)

        score = (
            self.w_conf * conf
            + self.w_nov * novelty
            - self.w_deg * deg_pen
            + self.w_rec * rec_bonus
        )

        if novelty == 0.0:
            action, reason = "aggregate", "duplicate_edge"
            if self.logger is not None:
                payload = {
                    "tuple": [head, rel, tail],
                    "deg_pen": deg_pen,
                    "conf": conf,
                    "score": score,
                }
                self.logger.log(mem="relational", action=action, reason=reason, payload=payload)
            return action, reason

        if score < self.threshold:
            action, reason = "route_to_episodic", f"score={score:.2f}<thr"
            if self.logger is not None:
                payload = {
                    "tuple": [head, rel, tail],
                    "deg_pen": deg_pen,
                    "conf": conf,
                    "score": score,
                }
                self.logger.log(mem="relational", action=action, reason=reason, payload=payload)
            return action, reason

        action, reason = "insert", f"score={score:.2f}>=thr"
        if self.logger is not None:
            payload = {
                "tuple": [head, rel, tail],
                "deg_pen": deg_pen,
                "conf": conf,
                "score": score,
            }
            self.logger.log(mem="relational", action=action, reason=reason, payload=payload)
        return action, reason


__all__ = ["RelationalGate"]
