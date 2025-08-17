"""Episodic write-gating utilities."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import faiss  # type: ignore
import numpy as np


@dataclass
class DGKey:
    """Sparse k-WTA encoded key used by the episodic store."""

    indices: np.ndarray
    values: np.ndarray
    dim: int


def k_wta(query: np.ndarray, k: int) -> DGKey:
    """Project ``query`` to a sparse key keeping the ``k`` largest magnitudes.

    Args:
        query: Dense input vector of shape ``(dim,)``.
        k: Number of winners to keep. ``k <= 0`` yields an empty key.
    """

    q = np.asarray(query, dtype="float32").reshape(-1)
    if k <= 0:
        return DGKey(
            indices=np.empty(0, dtype=np.int64),
            values=np.empty(0, dtype="float32"),
            dim=q.size,
        )
    k = min(k, q.size)
    idx = np.argpartition(-np.abs(q), k - 1)[:k]
    vals = q[idx]
    return DGKey(indices=idx.astype("int64"), values=vals.astype("float32"), dim=q.size)


def densify(key: DGKey) -> np.ndarray:
    """Convert a :class:`DGKey` back to a dense ``float32`` vector."""

    dense = np.zeros(key.dim, dtype="float32")
    dense[key.indices] = key.values
    return dense


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
    """Combine surprise, novelty, reward and pin signals into a write decision.

    The final salience score is computed as

    ``S = α·surprise + β·novelty + γ·reward + δ·pin``.
    """

    def __init__(
        self,
        tau: float = 0.5,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        delta: float = 1.0,
    ) -> None:
        """Create a ``WriteGate``.

        Args:
            tau: Threshold above which an item is written.
            alpha: Weight for the surprise term.
            beta: Weight for the novelty term.
            gamma: Weight for the reward term.
            delta: Weight for the pin term. Set ``delta=0`` to ignore ``pin``.
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
        pin: bool = False,
    ) -> float:
        """Return the combined salience score for a potential write."""

        s = surprise(prob)
        n = novelty(query, keys)
        return self.alpha * s + self.beta * n + self.gamma * reward + (self.delta if pin else 0.0)

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
        sc = self.score(prob, query, keys, reward, pin)
        return GateDecision(sc > self.tau, sc, provenance, timestamp)
