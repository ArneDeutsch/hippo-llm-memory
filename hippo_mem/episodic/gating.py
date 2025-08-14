"""Episodic write-gating utilities."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

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


@dataclass
class GateDecision:
    """Result of a write-gating decision."""

    allow: bool
    score: float
    provenance: str
    timestamp: float


class WriteGate:
    """Combine surprise, novelty and reward/pin signals into a write decision."""

    def __init__(
        self,
        tau: float = 0.5,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        delta: float = 0.0,
    ) -> None:
        """Create a ``WriteGate``.

        Args:
            tau: Threshold above which an item is written.
            alpha: Weight for the surprise term.
            beta: Weight for the novelty term.
            gamma: Weight for the reward term.
            delta: Bias added to the final score.
        """

        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def score(
        self,
        prob: float,
        query: np.ndarray,
        keys: np.ndarray,
        reward: float = 0.0,
    ) -> float:
        s = surprise(prob)
        n = novelty(query, keys)
        return self.alpha * s + self.beta * n + self.gamma * reward + self.delta

    def __call__(
        self,
        prob: float,
        query: np.ndarray,
        keys: np.ndarray,
        reward: float = 0.0,
        pin: bool = False,
        provenance: str = "",
        timestamp: float | None = None,
    ) -> GateDecision:
        if timestamp is None:
            timestamp = time.time()
        if pin:
            return GateDecision(True, float("inf"), provenance, timestamp)
        sc = self.score(prob, query, keys, reward)
        return GateDecision(sc > self.tau, sc, provenance, timestamp)
