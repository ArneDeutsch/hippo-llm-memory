"""Algorithm Card: HEI-NW Gating

Summary
-------
Combine neuromodulatory signals to decide when an episode should be written
to long-term storage.  Integration relies on a cross-attention adapter that
consumes recalled traces.

Integration style
-----------------
Cross-attention adapter wired after a transformer block.

Data structures
---------------
``DGKey`` sparse keys, ``TraceValue`` metadata payloads, ``AssocStore`` FAISS
index, ``ReplayQueue`` for prioritized consolidation.

Pipeline
--------
1. Encode residual stream -> ``DGKey`` via k-WTA.
2. Compute ``S = α·surprise + β·novelty + γ·reward + δ·pin``.
3. If ``S > τ`` persist (key, value) to ``AssocStore``.
4. Enqueue trace into ``ReplayQueue`` for CA2-style replay.

Design rationale & trade-offs
-----------------------------
Sparse keys limit interference; gating avoids store bloat.  Trade-off: missed
rare events when threshold ``τ`` too high.

Failure modes & diagnostics
---------------------------
High false negatives → inspect surprise/novelty stats; low recall → verify
key sparsity and index health.

Ablation switches & expected effects
------------------------------------
``use_gate=false`` writes all episodes, increasing interference.  ``replay.enabled=false``
prevents consolidation.

Contracts
---------
Persistence via FAISS + SQLite; gating score is a pure function ensuring
idempotent decisions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from hippo_mem.common import GateDecision, ProvenanceLogger, log_gate
from hippo_mem.common.telemetry import gate_registry

from .utils import cosine_dissimilarity

logger = logging.getLogger(__name__)


@dataclass
class DGKey:
    """Sparse k-WTA encoded key.

    Summary
    -------
    Represents the top-``k`` components of a dense vector.

    Parameters
    ----------
    indices : numpy.ndarray
        Winner indices with shape ``(k,)``.
    values : numpy.ndarray
        Winner magnitudes with shape ``(k,)``.
    dim : int
        Original dimensionality ``d``.
    Complexity
    ----------
    Storage ``O(k)``.

    Examples
    --------
    >>> key = DGKey(np.array([0, 2]), np.array([1.0, -0.5], dtype=np.float32), 4)
    >>> densify(key).shape
    (4,)

    See Also
    --------
    k_wta, densify
    """

    indices: np.ndarray
    values: np.ndarray
    dim: int


def k_wta(query: np.ndarray, k: int) -> DGKey:
    """Project ``query`` to a sparse key keeping the ``k`` largest magnitudes.

    Summary
    -------
    Implements k-winner-take-all encoding.

    Parameters
    ----------
    query : numpy.ndarray
        Dense input vector with shape ``(d,)``.
    k : int
        Number of winners; ``k <= 0`` returns an empty key.

    Returns
    -------
    DGKey
        Sparse representation of the top-``k`` elements.
    Complexity
    ----------
    ``O(d)`` for scanning ``query``.

    Examples
    --------
    >>> q = np.array([0.1, -0.4, 0.3], dtype=np.float32)
    >>> k_wta(q, 2).indices.shape
    (2,)

    See Also
    --------
    densify
    """

    q = np.asarray(query, dtype="float32").reshape(-1)
    if k <= 0:
        return DGKey(
            indices=np.empty(0, dtype=np.int64),
            values=np.empty(0, dtype="float32"),
            dim=q.size,
        )
    k = min(k, q.size)
    # why: argpartition finds the largest magnitudes without full sort
    idx = np.argpartition(-np.abs(q), k - 1)[:k]
    vals = q[idx]
    return DGKey(indices=idx.astype("int64"), values=vals.astype("float32"), dim=q.size)


def densify(key: DGKey) -> np.ndarray:
    """Convert a sparse key back to a dense vector.

    Summary
    -------
    Expand ``DGKey`` into a ``float32`` array of shape ``(d,)``.

    Parameters
    ----------
    key : DGKey
        Sparse key to expand.

    Returns
    -------
    numpy.ndarray
        Dense vector with zeros in non-winning positions.
    Complexity
    ----------
    ``O(d)`` to fill the dense array.

    Examples
    --------
    >>> key = DGKey(np.array([1]), np.array([0.5], dtype=np.float32), 3)
    >>> densify(key)
    array([0. , 0.5, 0. ], dtype=float32)

    See Also
    --------
    k_wta
    """

    dense = np.zeros(key.dim, dtype="float32")
    dense[key.indices] = key.values
    return dense


def surprise(prob: float) -> float:
    """Return the information content ``-log(p)`` of an event.

    Summary
    -------
    Larger values indicate higher surprise.

    Parameters
    ----------
    prob : float
        Probability of the event, ``0 < prob ≤ 1``.

    Returns
    -------
    float
        Information content in nats.
    Examples
    --------
    >>> round(surprise(0.5), 3)
    0.693

    See Also
    --------
    novelty
    """

    eps = 1e-8
    return -math.log(max(prob, eps))


def novelty(query: np.ndarray, keys: np.ndarray) -> float:
    """Compute novelty as ``1 - max_cos`` between ``query`` and stored keys.

    Summary
    -------
    Measures dissimilarity of ``query`` against catalogued keys.

    Parameters
    ----------
    query : numpy.ndarray
        Query embedding with shape ``(d,)``.
    keys : numpy.ndarray
        Stored key matrix with shape ``(n, d)``.

    Returns
    -------
    float
        Score in ``[0, 1]`` where ``1`` means unseen.
    Complexity
    ----------
    ``O(n d)`` for cosine similarity.

    Examples
    --------
    >>> novelty(np.zeros(2, dtype=np.float32), np.zeros((0, 2), dtype=np.float32))
    1.0

    See Also
    --------
    surprise
    """

    return cosine_dissimilarity(query, keys, "max")


class WriteGate:
    """Combine surprise, novelty, reward and pin signals into a write decision.

    Summary
    -------
    Implements ``S = α·surprise + β·novelty + γ·reward + δ·pin``.
    """

    def __init__(
        self,
        tau: float = 0.3,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        delta: float = 1.0,
        logger: ProvenanceLogger | None = None,
    ) -> None:
        """Initialise the gate.

        Parameters
        ----------
        tau : float, optional
            Threshold above which an item is written.
        alpha : float, optional
            Weight for the surprise term.
        beta : float, optional
            Weight for the novelty term.
        gamma : float, optional
            Weight for the reward term.
        delta : float, optional
            Weight for the pin term.
        Examples
        --------
        >>> gate = WriteGate(tau=0.1)
        >>> gate.tau
        0.1

        See Also
        --------
        score, __call__
        """

        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        for name, val in {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
        }.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.logger = logger

    def score(
        self,
        prob: float,
        query: np.ndarray,
        keys: np.ndarray,
        reward: float = 0.0,
        pin: bool = False,
    ) -> float:
        """Return the combined salience score for a potential write.

        Parameters
        ----------
        prob : float
            Model probability of the observed token.
        query : numpy.ndarray
            Query embedding of shape ``(d,)``.
        keys : numpy.ndarray
            Matrix of stored keys with shape ``(n, d)``.
        reward : float, optional
            External reward signal.
        pin : bool, optional
            User override to force writing.

        Returns
        -------
        float
            Combined salience ``S``.
        Complexity
        ----------
        ``O(n d)`` dominated by ``novelty``.

        Examples
        --------
        >>> gate = WriteGate()
        >>> round(gate.score(0.5, np.zeros(1), np.zeros((0, 1))), 3)
        0.346

        See Also
        --------
        novelty, surprise
        """

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
    ) -> GateDecision:
        """Decide whether to write an episode.

        Parameters
        ----------
        prob : float
            Model probability of the observed token.
        query : numpy.ndarray
            Query embedding of shape ``(d,)``.
        keys : numpy.ndarray
            Stored key matrix ``(n, d)``.
        reward : float, optional
            External reward signal.
        pin : bool, optional
            Force the decision irrespective of ``S``.
        provenance : str, optional
            Source identifier for logging.

        Returns
        -------
        GateDecision
            Decision record with computed score.
        Complexity
        ----------
        ``O(n d)`` via :meth:`score`.

        Examples
        --------
        >>> gate = WriteGate(tau=0.0)
        >>> gate(0.5, np.zeros(1), np.zeros((0, 1))).action
        'insert'

        See Also
        --------
        score
        """

        sc = self.score(prob, query, keys, reward, pin)
        allow = sc > self.tau
        action = "insert" if allow else "skip"
        reason = f"score={sc:.2f}"
        decision = GateDecision(action=action, reason=reason, score=sc)
        log_gate(
            self.logger,
            "episodic",
            decision,
            {"prob": prob, "provenance": provenance},
        )
        return decision


def gate_batch(
    gate: WriteGate,
    probs: np.ndarray,
    queries: np.ndarray,
    keys: np.ndarray,
    rewards: np.ndarray | None = None,
    pins: np.ndarray | None = None,
    provenance: str = "",
) -> tuple[list[GateDecision], float]:
    """Apply :class:`WriteGate` over a batch of items.

    Summary
    -------
    Compute decisions for each element and return the acceptance rate.

    Parameters
    ----------
    gate : WriteGate
        Gate used for scoring.
    probs : numpy.ndarray
        Model probabilities with shape ``(b,)``.
    queries : numpy.ndarray
        Query vectors of shape ``(b, d)``.
    keys : numpy.ndarray
        Existing key matrix ``(n, d)``.
    rewards : numpy.ndarray, optional
        Reward values ``(b,)``.
    pins : numpy.ndarray, optional
        Boolean overrides ``(b,)``.
    provenance : str, optional
        Source identifier for logging.

    Returns
    -------
    list of GateDecision
        Decision per batch element.
    float
        Fraction of accepted items.

    Examples
    --------
    >>> gate = WriteGate(tau=0.5)
    >>> decisions, rate = gate_batch(
    ...     gate,
    ...     np.array([0.1, 0.9]),
    ...     np.zeros((2, 1), dtype=np.float32),
    ...     np.zeros((0, 1), dtype=np.float32),
    ... )
    >>> (decisions[0].action, decisions[1].action, round(rate, 2))
    ('insert', 'skip', 0.5)
    """

    probs = np.asarray(probs, dtype=float).reshape(-1)
    queries = np.asarray(queries, dtype="float32").reshape(len(probs), -1)
    if rewards is not None:
        rewards = np.asarray(rewards, dtype=float).reshape(-1)
    if pins is not None:
        pins = np.asarray(pins, dtype=bool).reshape(-1)

    decisions: list[GateDecision] = []
    stats = gate_registry.get("episodic")
    if len(probs) == 0:
        stats.null_input += 1
        logger.info("write_accept_rate=0.00")
        return decisions, 0.0
    accepted = 0
    for i, p in enumerate(probs):
        stats.attempts += 1
        r = float(rewards[i]) if rewards is not None else 0.0
        pin = bool(pins[i]) if pins is not None else False
        dec = gate(p, queries[i], keys, r, pin, provenance)
        decisions.append(dec)
        if dec.action == "insert":
            stats.inserted += 1
            stats.accepted += 1
            accepted += 1
        else:
            stats.skipped += 1
    rate = accepted / len(decisions)
    logger.info("write_accept_rate=%.2f", rate)
    return decisions, rate
