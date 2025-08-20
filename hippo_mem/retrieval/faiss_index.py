"""FAISS index utilities.

Summary
-------
Wraps a tiny subset of the FAISS API with a NumPy fallback so tests run
without native extensions. The index stores vectors ``(d,)`` and supports
product quantisation.

See Also
--------
hippo_mem.retrieval.embed
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from hippo_mem._faiss import faiss


class FaissIndex:
    """Minimal FAISS wrapper with NumPy fallback.

    Summary
    -------
    Provides ``add``/``search`` over vectors ``(d,)`` using either
    :mod:`faiss` or a pure NumPy store.

    See Also
    --------
    embed_text
    """

    def __init__(self, dim: int, *, use_pq: bool = False, m: int = 8) -> None:
        """Create an empty index of dimension ``dim``.

        Summary
        -------
        Initialise either a FAISS index or a NumPy list based on availability.

        Parameters
        ----------
        dim : int
            Dimensionality of stored vectors.
        use_pq : bool, optional
            Use product quantisation when FAISS is available; default ``False``.
        m : int, optional
            Sub-quantizers for PQ; default ``8``.
        Side Effects
        ------------
        Allocates native FAISS structures when available.
        Examples
        --------
        >>> FaissIndex(4)
        <hippo_mem.retrieval.faiss_index.FaissIndex object...>

        See Also
        --------
        train
        """

        self.dim = dim
        self.use_pq = use_pq and faiss is not None
        if faiss is not None:
            if self.use_pq:
                self.index = faiss.IndexPQ(dim, m, 8)
            else:
                self.index = faiss.IndexFlatL2(dim)
        else:  # why: fallback keeps tests running without FAISS
            self._vectors: List[np.ndarray] = []

    # ------------------------------------------------------------------
    def train(self, data: Sequence[Sequence[float]]) -> None:
        """Train the underlying index if required.

        Summary
        -------
        No-op unless product quantisation is enabled.

        Parameters
        ----------
        data : sequence of sequence of float
            Training vectors ``(n, d)``.
        Side Effects
        ------------
        Updates FAISS internal state when PQ is used.

        Complexity
        ----------
        ``O(n d)`` for FAISS training.

        Examples
        --------
        >>> idx = FaissIndex(2, use_pq=True)
        >>> idx.train([[0.0, 0.0]])

        See Also
        --------
        add
        """

        if not self.use_pq or faiss is None:
            return  # nothing to do
        if self.index.is_trained:
            return
        mat = np.asarray(list(data), dtype="float32")
        if len(mat) == 0:
            return
        self.index.train(mat)

    def add(self, vector: Sequence[float]) -> None:
        """Add a single vector.

        Summary
        -------
        Inserts ``vector`` into FAISS or the fallback store.

        Parameters
        ----------
        vector : sequence of float
            Vector ``(d,)`` to index.
        Raises
        ------
        ValueError
            If ``vector`` has wrong dimensionality.

        Side Effects
        ------------
        Mutates the underlying index.

        Complexity
        ----------
        ``O(d)``.

        Examples
        --------
        >>> idx = FaissIndex(2)
        >>> idx.add([0.0, 0.1])

        See Also
        --------
        search
        """

        if len(vector) != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {len(vector)}")

        if faiss is not None:
            self.index.add(np.array([vector], dtype="float32"))
        else:
            self._vectors.append(np.array(vector, dtype="float32"))

    def search(self, query: Sequence[float], k: int = 1) -> List[int]:
        """Return indices of the nearest neighbours.

        Summary
        -------
        Finds ``k`` closest vectors to ``query`` using L2 distance.

        Parameters
        ----------
        query : sequence of float
            Query vector ``(d,)``.
        k : int, optional
            Number of neighbours to return; default ``1``.

        Returns
        -------
        list of int
            Indices of nearest neighbours.

        Raises
        ------
        ValueError
            If ``query`` dimensionality mismatches ``dim``.
        Complexity
        ----------
        ``O(n d)`` for fallback or FAISS search cost.

        Examples
        --------
        >>> idx = FaissIndex(2)
        >>> idx.add([0.0, 0.1])
        >>> idx.search([0.0, 0.1])
        [0]

        See Also
        --------
        add
        """

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
        """Remove the vector stored at ``idx``.

        Summary
        -------
        Deletes a vector from the underlying index or fallback list.

        Parameters
        ----------
        idx : int
            Position of vector to remove.
        Raises
        ------
        ValueError
            If ``idx`` is negative.
        IndexError
            If ``idx`` is out of range in fallback mode.

        Side Effects
        ------------
        Mutates the underlying index.

        Complexity
        ----------
        ``O(n)`` in fallback mode due to list deletion.

        Examples
        --------
        >>> idx = FaissIndex(2)
        >>> idx.add([0.0, 0.1])
        >>> idx.remove(0)

        See Also
        --------
        add
        """

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
        """Return the number of stored vectors.

        Summary
        -------
        Gives ``ntotal`` for FAISS or list length for fallback.
        Returns
        -------
        int
            Number of vectors in index.
        Examples
        --------
        >>> len(FaissIndex(2))
        0

        See Also
        --------
        add
        """

        if faiss is not None:
            return int(self.index.ntotal)
        return len(self._vectors)
