"""Schema prototypes and routing for SGC-RSS.

Summary
-------
Defines lightweight schema prototypes used to route tuples. High-score
tuples are written directly to the semantic graph; others are buffered
for episodic replay.
Complexity
----------
Scoring is ``O(#schemas)``.

Examples
--------
>>> from hippo_mem.relational.schema import SchemaIndex
>>> si = SchemaIndex(threshold=0.8)
>>> si.add_schema('buy', 'buy')
>>> si.score(si.schemas['buy'], ('A', 'buy', 'B', 'ctx', None, 1.0, 0))
1.0

See Also
--------
hippo_mem.relational.kg.KnowledgeGraph
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from .tuples import TupleType


def _pad_tuple(tup: TupleType | tuple) -> TupleType:
    """Return ``tup`` with default types when missing."""

    if len(tup) == 7:
        head, relation, tail, context, time, conf, prov = tup
        return (head, relation, tail, context, time, conf, prov, "entity", "entity")
    return tup  # type: ignore


if TYPE_CHECKING:  # pragma: no cover
    from .kg import KnowledgeGraph


@dataclass
class Schema:
    """Prototype describing expected tuple structure.

    Summary
    -------
    Captures relation name and optional type hints for entities.

    Parameters
    ----------
    name : str
        Human-readable schema identifier.
    relation : str
        Relation token expected in tuples.
    head_type : Optional[str], optional
        Anticipated type for the head entity.
    tail_type : Optional[str], optional
        Anticipated type for the tail entity.

    See Also
    --------
    SchemaIndex
    """

    name: str
    relation: str
    head_type: Optional[str] = None
    tail_type: Optional[str] = None


class SchemaIndex:
    """Store schema prototypes and route tuples based on confidence.

    Summary
    -------
    Maintains prototypes and buffers low-confidence tuples until they
    cross a threshold.

    Parameters
    ----------
    threshold : float, optional
        Minimum score and tuple confidence required for KG insertion.
    add_defaults : bool, optional
        When ``True`` preload simple schemas for synthetic data.
    """

    def __init__(self, threshold: float = 0.55, *, add_defaults: bool = True) -> None:
        self.schemas: Dict[str, Schema] = {}
        self.threshold = threshold
        self.episodic_buffer: List[TupleType] = []
        if add_defaults:
            for rel in ("bought", "bought_at", "is", "in", "located_in", "at"):
                self.add_schema(rel, rel)

    # kept: used by tests/test_relational.py and tests/test_replay_scheduler.py
    def add_schema(
        self,
        name: str,
        relation: str,
        head_type: Optional[str] = None,
        tail_type: Optional[str] = None,
    ) -> None:
        """Register a new schema prototype.

        Parameters
        ----------
        name, relation, head_type, tail_type
            Attributes for the schema; ``head_type`` and ``tail_type`` are
            optional type hints.
        """

        self.schemas[name] = Schema(name, relation, head_type, tail_type)

    def score(self, schema: Schema, tup: TupleType) -> float:
        """Return ``1.0`` when ``tup`` matches ``schema.relation``.

        Parameters
        ----------
        schema : Schema
            Prototype to evaluate.
        tup : TupleType
            Candidate tuple.

        Returns
        -------
        float
            Match confidence in ``[0, 1]``.
        """

        head, relation, tail, *_ = tup
        return 1.0 if relation == schema.relation else 0.0

    def fast_track(self, tup: TupleType, kg: KnowledgeGraph) -> bool:
        """Route ``tup`` to ``kg`` if confident; otherwise buffer.

        Summary
        -------
        Chooses the best schema score and inserts when both schema match
        and tuple confidence exceed ``threshold``.

        Parameters
        ----------
        tup : TupleType
            Tuple to evaluate.
        kg : KnowledgeGraph
            Target graph for insertion.

        Returns
        -------
        bool
            ``True`` if the tuple was written to ``kg``.
        """

        best_score = 0.0
        for schema in self.schemas.values():
            s = self.score(schema, tup)
            if s > best_score:
                best_score = s
        orig = tup
        tup = _pad_tuple(tup)
        if best_score >= self.threshold and tup[5] >= self.threshold:
            head, relation, tail, context, time, conf, prov, head_type, tail_type = tup
            kg.upsert(head, relation, tail, context, head_type, tail_type, time, conf, prov)
            # why: flushing handles newly qualified buffered tuples
            self.flush(kg)
            return True
        # why: keep low-confidence tuples for episodic replay
        self.episodic_buffer.append(orig)
        return False

    def flush(self, kg: KnowledgeGraph) -> None:
        """Promote buffered tuples meeting ``threshold`` to ``kg``.

        Parameters
        ----------
        kg : KnowledgeGraph
            Graph to receive promoted tuples.
        """

        remaining: List[TupleType] = []
        for tup in self.episodic_buffer:
            best_score = 0.0
            for schema in self.schemas.values():
                s = self.score(schema, tup)
                if s > best_score:
                    best_score = s
            padded = _pad_tuple(tup)
            if best_score >= self.threshold and padded[5] >= self.threshold:
                head, relation, tail, context, time, conf, prov, head_type, tail_type = padded
                kg.upsert(head, relation, tail, context, head_type, tail_type, time, conf, prov)
            else:
                remaining.append(tup)
        self.episodic_buffer = remaining


__all__ = ["SchemaIndex", "Schema"]
