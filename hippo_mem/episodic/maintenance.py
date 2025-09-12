# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .store import EpisodicStore


class Decayer:
    """Strategy for salience decay."""

    def decay(self, store: "EpisodicStore", rate: float) -> List[Any]:
        factor = max(0.0, 1.0 - rate)
        return store.persistence.decay(factor)


class Pruner:
    """Strategy for removing stale traces."""

    def prune(
        self, store: "EpisodicStore", min_salience: float, max_age: Optional[float]
    ) -> List[Any]:
        cutoff = time.time() - max_age if max_age is not None else None
        rows = store.persistence.fetch_prune_candidates(min_salience, cutoff)
        for row in rows:
            store.delete(int(row[0]))
        return rows


__all__ = ["Pruner", "Decayer"]
