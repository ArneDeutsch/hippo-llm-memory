"""Thread-safe retrieval telemetry counters."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict


@dataclass
class RetrievalStats:
    """Accumulated retrieval statistics."""

    requests: int = 0
    total_k: int = 0
    hits: int = 0
    tokens_returned: int = 0
    latency_ms_sum: float = 0.0

    def update(self, *, k: int, hits: int, tokens: int, latency_ms: float) -> None:
        """Add a retrieval observation."""

        self.requests += 1
        self.total_k += max(0, k)
        self.hits += max(0, hits)
        self.tokens_returned += max(0, tokens)
        self.latency_ms_sum += max(0.0, latency_ms)

    def snapshot(self) -> Dict[str, int | float]:
        """Return metrics with hit rate and average latency."""

        avg_latency = (self.latency_ms_sum / self.requests) if self.requests else 0.0
        hit_rate_at_k = (self.hits / self.total_k) if self.total_k else 0.0
        return {
            "requests": self.requests,
            "total_k": self.total_k,
            "hits": self.hits,
            "hit_rate_at_k": hit_rate_at_k,
            "tokens_returned": self.tokens_returned,
            "avg_latency_ms": avg_latency,
        }


class _Registry:
    """Thread-safe container for per-memory stats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, RetrievalStats] = {
            "episodic": RetrievalStats(),
            "relational": RetrievalStats(),
            "spatial": RetrievalStats(),
        }

    def get(self, name: str) -> RetrievalStats:
        """Return stats object for ``name``."""

        with self._lock:
            return self._stats[name]

    def reset(self) -> None:
        """Reset all counters to zero."""

        with self._lock:
            for key in self._stats:
                self._stats[key] = RetrievalStats()

    def all_snapshots(self) -> Dict[str, Dict[str, int | float]]:
        """Return snapshots for all registered memories."""

        with self._lock:
            return {k: v.snapshot() for k, v in self._stats.items()}


registry = _Registry()


def record_stats(kind: str, **metrics: int | float) -> None:
    """Update retrieval metrics for ``kind`` in the registry."""

    registry.get(kind).update(**metrics)


@dataclass
class GateStats:
    """Counters for gate decisions."""

    attempts: int = 0
    inserted: int = 0
    aggregated: int = 0
    routed_to_episodic: int = 0
    blocked_new_edges: int = 0

    def snapshot(self) -> Dict[str, int]:
        """Return raw gate counters."""

        accepts = self.inserted + self.aggregated
        return {
            "attempts": self.attempts,
            "accepts": accepts,
            "inserted": self.inserted,
            "aggregated": self.aggregated,
            "routed_to_episodic": self.routed_to_episodic,
            "blocked_new_edges": self.blocked_new_edges,
        }


class GateRegistry:
    """Thread-safe container for per-memory :class:`GateStats`."""

    def __init__(self, names: list[str]) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, GateStats] = {name: GateStats() for name in names}

    def get(self, name: str) -> GateStats:
        """Return stats object for ``name``."""

        with self._lock:
            return self._stats[name]

    def reset(self) -> None:
        """Reset all counters to zero."""

        with self._lock:
            for key in self._stats:
                self._stats[key] = GateStats()

    def all_snapshots(self) -> Dict[str, Dict[str, int]]:
        """Return snapshots for all registered memories."""

        with self._lock:
            return {k: v.snapshot() for k, v in self._stats.items()}


# Track gate counters for all memory types.
gate_registry = GateRegistry(["episodic", "relational", "spatial"])

__all__ = [
    "RetrievalStats",
    "registry",
    "record_stats",
    "GateStats",
    "GateRegistry",
    "gate_registry",
]
