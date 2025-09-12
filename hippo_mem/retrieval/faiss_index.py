# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Minimal vector index with FAISS or NumPy backends.

The module exposes a tiny ``add``/``search`` interface used by the tests. It
prefers a FAISS implementation when the bindings are available and otherwise
falls back to a pure NumPy variant so that CI runs without native extensions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from hippo_mem._faiss import faiss


class BaseIndex(ABC):
    """Common index protocol."""

    dim: int

    @abstractmethod
    def train(self, data: Sequence[Sequence[float]]) -> None:
        """Fit the index on ``data`` if required."""

    @abstractmethod
    def add(self, vector: Sequence[float]) -> None:
        """Insert ``vector`` into the index."""

    @abstractmethod
    def search(self, query: Sequence[float], k: int = 1) -> List[int]:
        """Return indices of the ``k`` nearest neighbours."""

    @abstractmethod
    def remove(self, idx: int) -> None:
        """Remove vector at ``idx``."""

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover - simple delegation
        """Number of stored vectors."""


class FaissBackend(BaseIndex):
    """Real FAISS implementation."""

    def __init__(self, dim: int, *, use_pq: bool, m: int) -> None:
        self.dim = dim
        self.use_pq = use_pq
        if use_pq:
            self.index = faiss.IndexPQ(dim, m, 8)
        else:
            self.index = faiss.IndexFlatL2(dim)

    def train(self, data: Sequence[Sequence[float]]) -> None:
        if not self.use_pq or self.index.is_trained:
            return
        mat = np.asarray(list(data), dtype="float32")
        if len(mat) > 0:
            self.index.train(mat)

    def add(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {len(vector)}")
        self.index.add(np.array([vector], dtype="float32"))

    def search(self, query: Sequence[float], k: int = 1) -> List[int]:
        if len(query) != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {len(query)}")
        if self.use_pq and not self.index.is_trained:
            return []
        _dists, idx = self.index.search(np.array([query], dtype="float32"), k)
        return [i for i in idx[0].tolist() if i != -1]

    def remove(self, idx: int) -> None:
        if idx < 0:
            raise ValueError("index must be non-negative")
        ids = np.array([idx], dtype="int64")
        removed = self.index.remove_ids(ids)
        if removed == 0:
            raise IndexError("index out of range")

    def __len__(self) -> int:
        return int(self.index.ntotal)


class NumpyBackend(BaseIndex):
    """Pure NumPy fallback used when FAISS is unavailable."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: List[np.ndarray] = []

    def train(self, data: Sequence[Sequence[float]]) -> None:
        return

    def add(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {len(vector)}")
        self._vectors.append(np.array(vector, dtype="float32"))

    def search(self, query: Sequence[float], k: int = 1) -> List[int]:
        if len(query) != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {len(query)}")
        if not self._vectors:
            return []
        mat = np.stack(self._vectors)
        dists = np.linalg.norm(mat - np.array(query, dtype="float32"), axis=1)
        return np.argsort(dists)[:k].tolist()

    def remove(self, idx: int) -> None:
        if idx < 0:
            raise ValueError("index must be non-negative")
        if idx < len(self._vectors):
            del self._vectors[idx]
        else:
            raise IndexError("index out of range")

    def __len__(self) -> int:
        return len(self._vectors)


class FaissIndex(BaseIndex):
    """User-facing wrapper selecting the appropriate backend."""

    def __init__(self, dim: int, *, use_pq: bool = False, m: int = 8) -> None:
        self.dim = dim
        self.use_pq = use_pq and faiss is not None
        if faiss is not None:
            self._backend: BaseIndex = FaissBackend(dim, use_pq=self.use_pq, m=m)
        else:  # fallback keeps tests running without FAISS
            self._backend = NumpyBackend(dim)

    def train(self, data: Sequence[Sequence[float]]) -> None:
        self._backend.train(data)

    def add(self, vector: Sequence[float]) -> None:
        self._backend.add(vector)

    def search(self, query: Sequence[float], k: int = 1) -> List[int]:
        return self._backend.search(query, k)

    def remove(self, idx: int) -> None:
        self._backend.remove(idx)

    def __len__(self) -> int:
        return len(self._backend)


__all__ = [
    "BaseIndex",
    "FaissBackend",
    "NumpyBackend",
    "FaissIndex",
]
