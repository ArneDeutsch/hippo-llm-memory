"""Index strategies for the episodic store."""

from __future__ import annotations

import logging
from typing import Dict, List, Protocol, Tuple

import numpy as np

from hippo_mem._faiss import faiss

logger = logging.getLogger(__name__)


class IndexStrategy(Protocol):
    """Protocol for vector index backends."""

    def add(self, key: np.ndarray, idx: int) -> None:
        """Add ``key`` with identifier ``idx``."""

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``k`` nearest neighbours for ``query``."""

    def remove(self, idx: int) -> None:
        """Remove vector with identifier ``idx``."""

    def update(self, key: np.ndarray, idx: int) -> None:
        """Replace vector at ``idx`` with ``key``."""

    @property
    def ntotal(self) -> int:
        """Number of stored vectors."""

    @property
    def is_trained(self) -> bool:
        """Whether the index is trained."""


class FaissIndex(IndexStrategy):
    """FAISS-backed index with optional training."""

    def __init__(self, dim: int, index_str: str, train_threshold: int) -> None:
        base = faiss.index_factory(dim, index_str, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base)
        self.train_threshold = train_threshold
        self._pending_keys: List[np.ndarray] = []
        self._pending_ids: List[int] = []

    def add(self, key: np.ndarray, idx: int) -> None:
        faiss.normalize_L2(key)
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

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        faiss.normalize_L2(query)
        return self.index.search(query, k)

    def remove(self, idx: int) -> None:
        try:
            self.remove_ids(np.array([idx], dtype="int64"))
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to remove id %s from index", idx)

    def remove_ids(self, ids: np.ndarray) -> None:
        """Expose ``remove_ids`` for tests expecting the old API."""
        self.index.remove_ids(ids)

    def update(self, key: np.ndarray, idx: int) -> None:
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


class NumpyIndex(IndexStrategy):
    """Simple in-memory index using NumPy for search."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vecs: Dict[int, np.ndarray] = {}

    def add(self, key: np.ndarray, idx: int) -> None:
        norm = np.linalg.norm(key, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        self._vecs[idx] = (key / norm)[0]

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._vecs:
            scores = np.zeros((1, k), dtype="float32")
            ids = np.full((1, k), -1, dtype="int64")
            return scores, ids
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        q = query / norm
        keys = np.stack(list(self._vecs.values()))
        ids = np.array(list(self._vecs.keys()), dtype="int64")
        sims = keys @ q.T
        sims = sims.reshape(-1)
        topk = sims.argsort()[::-1][:k]
        top_scores = sims[topk]
        top_ids = ids[topk]
        if len(topk) < k:
            pad = k - len(topk)
            top_scores = np.pad(top_scores, (0, pad))
            top_ids = np.pad(top_ids, (0, pad), constant_values=-1)
        return top_scores[np.newaxis, :], top_ids[np.newaxis, :]

    def remove(self, idx: int) -> None:
        self._vecs.pop(idx, None)

    def update(self, key: np.ndarray, idx: int) -> None:
        self._vecs[idx] = key[0]

    @property
    def ntotal(self) -> int:
        return len(self._vecs)

    @property
    def is_trained(self) -> bool:  # pragma: no cover - trivial
        return True


# Backwards compatibility
VectorIndex = FaissIndex


__all__ = ["IndexStrategy", "FaissIndex", "NumpyIndex", "VectorIndex"]
