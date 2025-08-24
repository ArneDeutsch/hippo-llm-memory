"""Common data structures and specifications for memory modules."""

from .gates import GateDecision, GateResult, MemoryGate
from .provenance import ProvenanceLogger, log_gate
from .specs import MemoryTokens, TraceSpec

__all__ = [
    "MemoryTokens",
    "TraceSpec",
    "ProvenanceLogger",
    "log_gate",
    "GateDecision",
    "GateResult",
    "MemoryGate",
]
