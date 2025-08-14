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
    """Very small wrapper around :mod:`faiss` or a Python fallback."""

    def __init__(self, dim: int) -> None:
        """Create an empty L2 index of dimension ``dim``.

        Parameters
        ----------
        dim:
            Dimensionality of the vectors that will be stored in the index.
        """

        self.dim = dim
        if faiss is not None:
            self.index = faiss.IndexFlatL2(dim)
        else:  # fallback to a very small numpy based index
            self._vectors: List[np.ndarray] = []

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
            _dists, idx = self.index.search(np.array([query], dtype="float32"), k)
            return idx[0].tolist()

        if not self._vectors:
            return []

        mat = np.stack(self._vectors)
        dists = np.linalg.norm(mat - np.array(query, dtype="float32"), axis=1)
        return np.argsort(dists)[:k].tolist()
