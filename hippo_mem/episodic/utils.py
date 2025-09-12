# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Utility functions for episodic memory."""

from __future__ import annotations

import numpy as np

from hippo_mem._faiss import faiss


def cosine_dissimilarity(vec: np.ndarray, matrix: np.ndarray, reduce: str) -> float:
    """Return ``1 - cos(vec, matrix)`` under a reduction.

    Parameters
    ----------
    vec : numpy.ndarray
        Query vector with shape ``(d,)``.
    matrix : numpy.ndarray
        Matrix of stored vectors ``(n, d)``.
    reduce : str
        Reduction method: ``"max"`` or ``"mean"``.

    Returns
    -------
    float
        Dissimilarity score in ``[0, 1]``.
    """

    if matrix.size == 0:
        return 1.0

    q = np.asarray(vec, dtype="float32").reshape(1, -1)
    m = np.asarray(matrix, dtype="float32")
    faiss.normalize_L2(q)
    faiss.normalize_L2(m)
    sims = m @ q[0]
    if reduce == "max":
        sim = float(np.max(sims))
    elif reduce == "mean":
        sim = float(np.mean(sims))
    else:
        raise ValueError("reduce must be 'max' or 'mean'")
    return 1.0 - sim
