"""Common data structures and specifications for memory modules."""

from .attn_adapter import CrossAttnAdapter, LoraLinear
from .gates import GateDecision, GateResult, MemoryGate
from .history import HistoryEntry, RollbackMixin
from .lifecycle import StoreLifecycleMixin
from .maintenance import BackgroundTaskManager
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
    "BackgroundTaskManager",
    "StoreLifecycleMixin",
    "HistoryEntry",
    "RollbackMixin",
    "CrossAttnAdapter",
    "LoraLinear",
]
