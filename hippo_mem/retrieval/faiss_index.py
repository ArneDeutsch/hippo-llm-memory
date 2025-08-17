"""Shared ANN fabric with optional FAISS backend.

Summary
-------
Wraps a minimal subset of the FAISS API for embedding-based retrieval used by
episodic, semantic, and spatial stores.  Consolidation batches rely on this
fabric for the 50/30/20 replay mix and background maintenance threads prune or
merge entries to control growth.

See Also
--------
hippo_mem.retrieval.embed
    Deterministic placeholder embedder.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

try:  # pragma: no cover - import side effects only
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback path
    faiss = None


class FaissIndex:
    """Minimal index abstraction with optional product quantisation.

    Summary
    -------
    Provides a uniform API over a FAISS backend or a NumPy fallback so tests
    can run without native dependencies.

    Parameters
    ----------
    dim:
        Dimensionality of stored vectors.
    use_pq:
        Whether to enable product quantisation (requires training).
    m:
        Number of sub-quantisers when PQ is enabled.

    Returns
    -------
    None

    Raises
    ------
    None

    Side Effects
    ------------
    Allocates FAISS structures if available.

    Complexity
    ----------
    Index construction is ``O(1)``; training and queries depend on backend.

    Examples
    --------
    >>> idx = FaissIndex(4)
    >>> len(idx)
    0

    See Also
    --------
    train
    add
    search
    """

    def __init__(self, dim: int, *, use_pq: bool = False, m: int = 8) -> None:
        """Create an empty index of dimension ``dim``.

        Summary
        -------
        Initialises either a FAISS index or a Python fallback.

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

        Returns
        -------
        None

        Raises
        ------
        None

        Side Effects
        ------------
        Allocates index structures.

        Complexity
        ----------
        ``O(1)``.

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
        else:  # fallback to a very small numpy based index
            self._vectors: List[np.ndarray] = []

    # ------------------------------------------------------------------
    def train(self, data: Sequence[Sequence[float]]) -> None:
        """Train the underlying index on ``data`` if required.

        Summary
        -------
        Only relevant when product quantisation is enabled.

        Parameters
        ----------
        data:
            Iterable of vectors with shape ``(N, D)``.

        Returns
        -------
        None

        Raises
        ------
        None

        Side Effects
        ------------
        Mutates the FAISS index state.

        Complexity
        ----------
        ``O(ND)`` for training matrix ``N`` by ``D``.

        Examples
        --------
        >>> idx = FaissIndex(2, use_pq=True)
        >>> idx.train([[0.0, 0.0]])

        See Also
        --------
        __init__
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
        """Add a single vector to the index.

        Summary
        -------
        Stores the vector in either the FAISS backend or the Python list.

        Parameters
        ----------
        vector:
            Sequence of length ``dim``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the vector has the wrong dimensionality.

        Side Effects
        ------------
        Mutates the index storage.

        Complexity
        ----------
        ``O(1)``.

        Examples
        --------
        >>> idx = FaissIndex(2)
        >>> idx.add([0.0, 0.1])

        See Also
        --------
        search
        remove
        """

        if len(vector) != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {len(vector)}")

        if faiss is not None:
            self.index.add(np.array([vector], dtype="float32"))
        else:
            self._vectors.append(np.array(vector, dtype="float32"))

    def search(self, query: Sequence[float], k: int = 1) -> List[int]:
        """Return indices of the ``k`` nearest neighbours for ``query``.

        Summary
        -------
        Performs an exact or approximate search depending on backend.

        Parameters
        ----------
        query:
            Query vector of length ``dim``.
        k:
            Number of neighbours to retrieve.

        Returns
        -------
        list[int]
            Indices of nearest neighbours.

        Raises
        ------
        ValueError
            If the query has the wrong dimensionality.

        Side Effects
        ------------
        None

        Complexity
        ----------
        FAISS: ``O(log n)``; fallback: ``O(n)`` for ``n`` vectors.

        Examples
        --------
        >>> idx = FaissIndex(2)
        >>> idx.add([0.0, 0.1])
        >>> idx.search([0.0, 0.1], k=1)
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
        """Remove the vector stored at ``idx`` from the index.

        Summary
        -------
        Deletes the entry, enabling maintenance jobs such as pruning.

        Parameters
        ----------
        idx:
            Position of the vector to remove.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``idx`` is negative.
        IndexError
            If ``idx`` is out of range for the fallback storage.

        Side Effects
        ------------
        Mutates storage in-place.

        Complexity
        ----------
        ``O(1)``.

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
        """Return the number of vectors currently stored.

        Summary
        -------
        Provides a unified way to inspect storage utilisation.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of stored vectors.

        Raises
        ------
        None

        Side Effects
        ------------
        None

        Complexity
        ----------
        ``O(1)``.

        Examples
        --------
        >>> len(FaissIndex(2))
        0

        See Also
        --------
        add
        remove
        """

        if faiss is not None:
            return int(self.index.ntotal)
        return len(self._vectors)
