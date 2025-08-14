"""Episodic write-gating utilities."""

from __future__ import annotations

import math

import faiss  # type: ignore
import numpy as np


def surprise(prob: float) -> float:
    """Return the information content ``-log(p)`` of an event."""

    eps = 1e-8
    return -math.log(max(prob, eps))


def novelty(query: np.ndarray, keys: np.ndarray) -> float:
    """Compute novelty as ``1 - max_cos`` between ``query`` and stored ``keys``.

    Args:
        query: Query embedding of shape ``(dim,)``.
        keys: Array of stored keys with shape ``(n, dim)``.

    Returns:
        A float in ``[0, 1]`` where ``1`` means completely novel.
    """

    if keys.size == 0:
        return 1.0

    q = np.asarray(query, dtype="float32").reshape(1, -1)
    k = np.asarray(keys, dtype="float32")
    faiss.normalize_L2(q)
    faiss.normalize_L2(k)
    sims = k @ q[0]
    return 1.0 - float(np.max(sims))


class WriteGate:
    """Combine surprise, novelty and reward/pin signals into a write decision."""

    def __init__(self, tau: float = 0.5) -> None:
        self.tau = tau

    def score(
        self,
        prob: float,
        query: np.ndarray,
        keys: np.ndarray,
        reward: float = 0.0,
    ) -> float:
        s = surprise(prob)
        n = novelty(query, keys)
        return 0.5 * (s + n) + reward

    def __call__(
        self,
        prob: float,
        query: np.ndarray,
        keys: np.ndarray,
        reward: float = 0.0,
        pin: bool = False,
    ) -> tuple[bool, float]:
        if pin:
            return True, float("inf")
        sc = self.score(prob, query, keys, reward)
        return sc > self.tau, sc
