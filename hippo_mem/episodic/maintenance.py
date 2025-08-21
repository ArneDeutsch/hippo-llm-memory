"""Pruning and decay strategies for episodic stores."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

if False:
    from .store import EpisodicStore  # pragma: no cover


@dataclass
class Decayer:
    """Exponential salience decay policy."""

    rate: float = 0.0

    def decay(self, store: "EpisodicStore") -> None:
        if self.rate <= 0:
            return
        factor = max(0.0, 1.0 - self.rate)
        rows = store.db.decay(factor)
        if rows:
            store._push_history("decay", rows)
        store.logger.log("decay", {"rate": self.rate})


@dataclass
class Pruner:
    """Remove stale or low-salience memories."""

    min_salience: float = 0.1
    max_age: Optional[float] = None

    def prune(self, store: "EpisodicStore") -> None:
        cutoff = time.time() - self.max_age if self.max_age is not None else None
        rows = store.db.fetch_prune_candidates(self.min_salience, cutoff)
        if not rows:
            return
        store._push_history("prune", rows)
        for row in rows:
            store.delete(int(row[0]))
        store.logger.log("prune", {"min_salience": self.min_salience, "max_age": self.max_age})


__all__ = ["Pruner", "Decayer"]
