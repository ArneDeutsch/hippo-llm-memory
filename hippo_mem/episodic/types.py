"""Dataclasses for episodic traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .gating import DGKey


@dataclass
class TraceValue:
    """Metadata associated with a stored trace."""

    tokens_span: Optional[tuple[int, int]] = None
    entity_slots: Optional[dict] = None
    state_sketch: Optional[list] = None
    salience_tags: Optional[List[str]] = None
    provenance: Optional[str] = None
    trace_id: Optional[str] = None
    sample_id: Optional[str] = None
    suite: Optional[str] = None


@dataclass
class Trace:
    """A retrieved memory trace."""

    id: int
    value: TraceValue
    key: DGKey
    score: float
    ts: float
    salience: float


__all__ = ["TraceValue", "Trace"]
