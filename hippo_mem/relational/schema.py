"""Simple schema index for routing relational tuples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from .tuples import TupleType

if TYPE_CHECKING:  # pragma: no cover
    from .kg import KnowledgeGraph


@dataclass
class Schema:
    name: str
    relation: str
    head_type: Optional[str] = None
    tail_type: Optional[str] = None


class SchemaIndex:
    """Store schema prototypes and route tuples based on confidence."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.schemas: Dict[str, Schema] = {}
        self.threshold = threshold
        self.episodic_buffer: List[TupleType] = []

    def add_schema(
        self,
        name: str,
        relation: str,
        head_type: Optional[str] = None,
        tail_type: Optional[str] = None,
    ) -> None:
        self.schemas[name] = Schema(name, relation, head_type, tail_type)

    def score(self, schema: Schema, tup: TupleType) -> float:
        head, relation, tail, *_ = tup
        return 1.0 if relation == schema.relation else 0.0

    def fast_track(self, tup: TupleType, kg: KnowledgeGraph) -> bool:
        """Route ``tup`` to ``kg`` if confident enough, else keep in episodic buffer."""

        best_score = 0.0
        for schema in self.schemas.values():
            s = self.score(schema, tup)
            if s > best_score:
                best_score = s
        if best_score >= self.threshold and tup[5] >= self.threshold:
            head, relation, tail, context, time, conf, prov = tup
            kg.upsert(head, relation, tail, context, time, conf, prov)
            self.flush(kg)
            return True
        self.episodic_buffer.append(tup)
        return False

    def flush(self, kg: KnowledgeGraph) -> None:
        """Attempt to write buffered tuples to ``kg`` if they now meet the threshold."""

        remaining: List[TupleType] = []
        for tup in self.episodic_buffer:
            best_score = 0.0
            for schema in self.schemas.values():
                s = self.score(schema, tup)
                if s > best_score:
                    best_score = s
            if best_score >= self.threshold and tup[5] >= self.threshold:
                head, relation, tail, context, time, conf, prov = tup
                kg.upsert(head, relation, tail, context, time, conf, prov)
            else:
                remaining.append(tup)
        self.episodic_buffer = remaining


__all__ = ["SchemaIndex", "Schema"]
