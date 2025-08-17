"""FAISS helpers for the episodic store."""

from __future__ import annotations

import logging
from typing import List

import faiss  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    """Thin wrapper around a FAISS index with ID management."""

    def __init__(self, dim: int, index_str: str, train_threshold: int) -> None:
        base = faiss.index_factory(dim, index_str, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base)
        self.train_threshold = train_threshold
        self._pending_keys: List[np.ndarray] = []
        self._pending_ids: List[int] = []

    # ------------------------------------------------------------------
    def add(self, key: np.ndarray, idx: int) -> None:
        """Add ``key`` with identifier ``idx`` training the index if needed."""

        if self.index.is_trained:
            ids = np.array([idx], dtype="int64")
            self.index.add_with_ids(key, ids)
        else:
            self._pending_keys.append(key[0])
            self._pending_ids.append(idx)
            if len(self._pending_keys) >= self.train_threshold:
                train_mat = np.vstack(self._pending_keys)
                faiss.normalize_L2(train_mat)
                self.index.train(train_mat)
                ids = np.array(self._pending_ids, dtype="int64")
                self.index.add_with_ids(train_mat, ids)
                self._pending_keys.clear()
                self._pending_ids.clear()

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``k`` nearest neighbours for ``query``."""

        return self.index.search(query, k)

    def remove(self, idx: int) -> None:
        """Remove vector with identifier ``idx`` from the index."""

        try:
            self.remove_ids(np.array([idx], dtype="int64"))
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to remove id %s from index", idx)

    def remove_ids(self, ids: np.ndarray) -> None:
        """Expose ``remove_ids`` for compatibility with tests."""

        self.index.remove_ids(ids)

    def update(self, key: np.ndarray, idx: int) -> None:
        """Replace vector at ``idx`` with ``key`` (handles untrained indices)."""

        self.remove(idx)
        if self.index.is_trained:
            self.add(key, idx)
        else:
            self._pending_keys.append(key[0])
            self._pending_ids.append(idx)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @property
    def is_trained(self) -> bool:
        return bool(self.index.is_trained)


__all__ = ["VectorIndex"]
