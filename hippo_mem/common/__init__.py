# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Common data structures and specifications for memory modules."""

from .attn_adapter import CrossAttnAdapter, CrossAttnConfig, LoraLinear
from .gates import GateDecision, GateResult, MemoryGate
from .history import HistoryEntry, RollbackMixin
from .io import (
    atomic_write_file,
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    read_parquet,
)
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
    "CrossAttnConfig",
    "LoraLinear",
    "atomic_write_file",
    "atomic_write_json",
    "atomic_write_jsonl",
    "read_json",
    "read_jsonl",
    "read_parquet",
]
