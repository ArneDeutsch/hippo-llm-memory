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

__all__ = ["RetrievalStats", "registry"]
