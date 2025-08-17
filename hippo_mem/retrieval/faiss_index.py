"""FAISS index utilities.

This module wraps a tiny subset of the FAISS API that we rely on.  FAISS is an
optional dependency; when it is not available we fall back to a simple
``numpy`` based implementation so that tests can run without native extensions.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

try:  # pragma: no cover - import side effects only
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback path
    faiss = None


class FaissIndex:
    """Very small wrapper around :mod:`faiss` or a Python fallback.

    Only the bits of the API that are exercised in the unit tests are
    implemented.  When :mod:`faiss` is not available the class falls back to a
    tiny ``numpy`` based index so that tests can run without native
    dependencies.
    """

    def __init__(self, dim: int, *, use_pq: bool = False, m: int = 8) -> None:
        """Create an empty index of dimension ``dim``.

        Parameters
        ----------
        dim:
            Dimensionality of the vectors that will be stored in the index.
        use_pq:
            If ``True`` and FAISS is available an ``IndexPQ`` will be used which
            requires an explicit training step.
        m:
            Number of sub-quantizers for product quantisation when ``use_pq`` is
            enabled.  The default keeps things small for tests.
        """

        self.dim = dim
        self.use_pq = use_pq and faiss is not None
        if faiss is not None:
            if self.use_pq:
                self.index = faiss.IndexPQ(dim, m, 8)
            else:
                self.index = faiss.IndexFlatL2(dim)
        else:  # fallback to a very small numpy based index
            self._vectors: List[np.ndarray] = []

    # ------------------------------------------------------------------
    def train(self, data: Sequence[Sequence[float]]) -> None:
        """Train the underlying index on ``data`` if required."""

        if not self.use_pq or faiss is None:
            return  # nothing to do
        if self.index.is_trained:
            return
        mat = np.asarray(list(data), dtype="float32")
        if len(mat) == 0:
            return
        self.index.train(mat)

    def add(self, vector: Sequence[float]) -> None:
        """Add a single vector to the index."""

        if len(vector) != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {len(vector)}")

        if faiss is not None:
            self.index.add(np.array([vector], dtype="float32"))
        else:
            self._vectors.append(np.array(vector, dtype="float32"))

    def search(self, query: Sequence[float], k: int = 1) -> List[int]:
        """Return indices of the ``k`` nearest neighbours for ``query``."""

        if len(query) != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {len(query)}")

        if faiss is not None:
            if self.use_pq and not self.index.is_trained:
                return []
            _dists, idx = self.index.search(np.array([query], dtype="float32"), k)
            return [i for i in idx[0].tolist() if i != -1]

        if not self._vectors:
            return []

        mat = np.stack(self._vectors)
        dists = np.linalg.norm(mat - np.array(query, dtype="float32"), axis=1)
        return np.argsort(dists)[:k].tolist()

    def remove(self, idx: int) -> None:
        """Remove the vector stored at ``idx`` from the index."""

        if idx < 0:
            raise ValueError("index must be non-negative")

        if faiss is not None:
            ids = np.array([idx], dtype="int64")
            self.index.remove_ids(ids)
        elif idx < len(self._vectors):
            del self._vectors[idx]
        else:
            raise IndexError("index out of range")

    def __len__(self) -> int:
        """Return the number of vectors currently stored."""

        if faiss is not None:
            return int(self.index.ntotal)
        return len(self._vectors)
