# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Prioritised replay scheduler.

Summary
-------
Implements CA2-style replay mixing 50% episodic, 30% semantic and 20% fresh
items per batch. Ordering combines salience, recency and diversity to limit
interference. Maintenance jobs such as decay, prune and merge are executed by
the consolidation worker.

See Also
--------
hippo_mem.consolidation.worker
"""

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph

from .utils import cosine_dissimilarity


@dataclass
class ReplayItem:
    """Metadata stored for replay scheduling.

    Summary
    -------
    Captures attributes required for priority computation.

    Parameters
    ----------
    trace_id : str
        Identifier of the trace in the store.
    key : numpy.ndarray
        Dense key vector ``(d,)``.
    score : float
        Gating score ``S``.
    step : int
        Global step when inserted.
    diversity_sig : float
        Diversity signature in ``[0, 1]``.
    grad_overlap_proxy : float, optional
        Approximation of gradient overlap.
    grad : numpy.ndarray, optional
        Gradient estimate ``(d,)``.
    timestamp : float
        Event time in seconds.
    Examples
    --------
    >>> ReplayItem("t", np.zeros(1), 0.0, 0, 1.0, None, None, 0.0).trace_id
    't'

    See Also
    --------
    ReplayQueue
    """

    trace_id: str
    key: np.ndarray
    score: float
    step: int
    diversity_sig: float
    grad_overlap_proxy: Optional[float]
    grad: Optional[np.ndarray]
    timestamp: float


class ReplayQueue:
    """Replay queue mixing gating score, recency and diversity.

    Summary
    -------
    Maintains ``ReplayItem`` objects sorted by a weighted priority to achieve
    interference-aware ordering.

    See Also
    --------
    ReplayScheduler
    """

    def __init__(
        self,
        maxlen: int = 1024,
        *,
        lambda1: float = 0.5,
        lambda2: float = 0.3,
        lambda3: float = 0.2,
    ) -> None:
        """Initialise an empty priority queue.

        Summary
        -------
        Sets weighting coefficients for salience (λ₁), recency (λ₂) and
        diversity (λ₃).

        Parameters
        ----------
        maxlen : int, optional
            Maximum queue length; default ``1024``.
        lambda1 : float, optional
            Weight for salience component; default ``0.5``.
        lambda2 : float, optional
            Weight for recency; default ``0.3``.
        lambda3 : float, optional
            Weight for diversity; default ``0.2``.
        Examples
        --------
        >>> ReplayQueue(maxlen=10).maxlen
        10

        See Also
        --------
        add
        """

        self.maxlen = maxlen
        self.items: List[ReplayItem] = []
        self.step = 0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def _diversity(self, key: np.ndarray, keys: np.ndarray) -> float:
        """Summary
        -------
        Compute average cosine dissimilarity between ``key`` and ``keys`` to
        favour novel items.

        Parameters
        ----------
        key : numpy.ndarray
            Candidate key ``(d,)``.
        keys : numpy.ndarray
            Existing key matrix ``(n, d)``.

        Returns
        -------
        float
            Diversity score in ``[0, 1]``.
        Complexity
        ----------
        ``O(n d)``.

        Examples
        --------
        >>> ReplayQueue()._diversity(np.zeros(1), np.zeros((0, 1)))
        1.0

        See Also
        --------
        _priority_scores
        """

        return cosine_dissimilarity(key, keys, "mean")

    def add(
        self,
        trace_id: str,
        key: np.ndarray,
        score: float,
        *,
        grad_overlap_proxy: Optional[float] = None,
        grad: Optional[np.ndarray] = None,
        timestamp: float | None = None,
    ) -> None:
        """Summary
        -------
        Add a trace with associated key and gating score ``S``.

        Parameters
        ----------
        trace_id : str
            Identifier of the trace.
        key : numpy.ndarray
            Key vector ``(d,)``.
        score : float
            Gating score ``S``.
        grad_overlap_proxy : float, optional
            Proxy for gradient overlap.
        grad : numpy.ndarray, optional
            Gradient estimate ``(d,)``.
        timestamp : float, optional
            Event time in seconds.
        Side Effects
        ------------
        Mutates the queue in place.

        Complexity
        ----------
        ``O(n d)`` due to diversity computation.

        Examples
        --------
        >>> q = ReplayQueue()
        >>> q.add("t", np.zeros(1), 0.0)
        >>> len(q.items)
        1

        See Also
        --------
        sample
        """

        self.step += 1
        if timestamp is None:
            timestamp = time.time()
        k = np.asarray(key, dtype="float32")
        keys = (
            np.stack([it.key for it in self.items])
            if self.items
            else np.empty((0, k.size), dtype="float32")
        )
        diversity = self._diversity(k, keys)
        item = ReplayItem(
            trace_id=trace_id,
            key=k,
            score=float(score),
            step=self.step,
            diversity_sig=diversity,
            grad_overlap_proxy=grad_overlap_proxy,
            grad=np.asarray(grad, dtype="float32") if grad is not None else None,
            timestamp=timestamp,
        )
        self.items.append(item)
        if len(self.items) > self.maxlen:
            self.items.pop(0)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity between vectors.

        Summary
        -------
        Safe cosine computation avoiding division by zero.

        Parameters
        ----------
        a, b : numpy.ndarray
            Vectors ``(d,)``.

        Returns
        -------
        float
            Cosine similarity ``[-1, 1]``.
        Complexity
        ----------
        ``O(d)``.

        Examples
        --------
        >>> ReplayQueue()._cosine(np.ones(1), np.ones(1))
        1.0

        See Also
        --------
        _diversity
        """

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _priority_scores(self) -> List[float]:
        """Return priority scores for the current queue items.

        Summary
        -------
        Combines gating score, recency and diversity components.
        Returns
        -------
        list of float
            Priority values aligned with ``self.items``.
        Complexity
        ----------
        ``O(n)``.

        Examples
        --------
        >>> q = ReplayQueue()
        >>> q._priority_scores()
        []

        See Also
        --------
        _select_indices
        """

        now = self.step
        priorities: List[float] = []
        for it in self.items:
            recency = 1.0 / (now - it.step + 1)
            diversity_component = it.diversity_sig + (it.grad_overlap_proxy or 0.0)
            p = (
                self.lambda1 * it.score
                + self.lambda2 * recency
                + self.lambda3 * diversity_component
            )
            priorities.append(p)
        return priorities

    def _select_indices(self, order: List[int], k: int, grad_sim_threshold: float) -> List[int]:
        """Select indices respecting gradient overlap constraints.

        Summary
        -------
        Greedy selection avoiding highly similar gradients.

        Parameters
        ----------
        order : list of int
            Candidate indices sorted by priority.
        k : int
            Number of items to pick.
        grad_sim_threshold : float
            Maximum allowed cosine similarity.

        Returns
        -------
        list of int
            Selected indices.
        Complexity
        ----------
        ``O(k d)``.

        Examples
        --------
        >>> q = ReplayQueue()
        >>> q._select_indices([], 1, 0.9)
        []

        See Also
        --------
        sample
        """

        selected: List[int] = []
        last_grad: Optional[np.ndarray] = None
        for idx in order:
            if len(selected) >= k:
                break
            item = self.items[idx]
            if last_grad is not None and item.grad is not None:
                if self._cosine(last_grad, item.grad) >= grad_sim_threshold:
                    # why: skip traces with similar gradients to preserve diversity
                    continue
            selected.append(idx)
            if item.grad is not None:
                last_grad = item.grad
        return selected

    def sample(self, k: int, *, grad_sim_threshold: float = 0.9) -> List[str]:
        """Summary
        -------
        Return ``k`` trace identifiers prioritising low gradient overlap.

        Parameters
        ----------
        k : int
            Number of traces.
        grad_sim_threshold : float, optional
            Maximum allowed cosine similarity.

        Returns
        -------
        list of str
            Selected trace identifiers.
        Complexity
        ----------
        ``O(n log n)`` due to sorting.

        Examples
        --------
        >>> q = ReplayQueue()
        >>> q.sample(1)
        []

        See Also
        --------
        add
        """

        if not self.items:
            return []
        priorities = self._priority_scores()
        order = list(np.argsort(priorities)[::-1])
        selected = self._select_indices(order, k, grad_sim_threshold)
        return [self.items[i].trace_id for i in selected]


class BatchMixLike(Protocol):
    """Protocol describing the batch mix structure.

    Summary
    -------
    Exposes proportions of replay types.
    """

    episodic: float
    semantic: float
    fresh: float  # kept: scheduler derives fresh from remainder; see tests/test_replay_scheduler.py


class ReplayScheduler:
    """Scheduler that interleaves episodic, semantic and fresh items.

    Summary
    -------
    Wraps a :class:`ReplayQueue` and knowledge graph to produce replay batches.
    Default mix is 50% episodic, 30% semantic and 20% fresh; metrics are logged
    for diagnostics.

    See Also
    --------
    ConsolidationWorker
    """

    def __init__(
        self,
        store: EpisodicStore,
        kg: KnowledgeGraph,
        *,
        batch_mix: BatchMixLike,
        grad_sim_threshold: float = 0.9,
    ) -> None:
        """Summary
        -------
        Initialise scheduler state with replay ratios and gradient threshold.

        Parameters
        ----------
        store : EpisodicStore
            Backing episodic store.
        kg : KnowledgeGraph
            Semantic store for relational recalls.
        batch_mix : BatchMixLike
            Ratios of replay types.
        grad_sim_threshold : float, optional
            Gradient overlap threshold.
        Examples
        --------
        >>> from hippo_mem.relational.kg import KnowledgeGraph
        >>> scheduler = ReplayScheduler(EpisodicStore(1), KnowledgeGraph(), batch_mix=type('B',(),{'episodic':1.0,'semantic':0.0,'fresh':0.0})())

        See Also
        --------
        next_batch
        """

        self.store = store
        self.kg = kg
        self.batch_mix = batch_mix
        self.queue = ReplayQueue()
        self.grad_sim_threshold = grad_sim_threshold
        self._log = {"batches": 0}

    def add_trace(
        self,
        trace_id: str,
        key: np.ndarray,
        score: float,
        *,
        grad_overlap_proxy: Optional[float] = None,
        grad: Optional[np.ndarray] = None,
        timestamp: float | None = None,
    ) -> None:
        """Summary
        -------
        Proxy for :meth:`ReplayQueue.add` to enqueue traces.

        Parameters
        ----------
        trace_id : str
            Identifier of the trace.
        key : numpy.ndarray
            Key vector ``(d,)``.
        score : float
            Gating score ``S``.
        grad_overlap_proxy : float, optional
            Gradient overlap proxy.
        grad : numpy.ndarray, optional
            Gradient estimate ``(d,)``.
        timestamp : float, optional
            Event time in seconds.
        Side Effects
        ------------
        Enqueues item into internal queue.

        Complexity
        ----------
        As in :meth:`ReplayQueue.add`.

        Examples
        --------
        >>> scheduler = ReplayScheduler(EpisodicStore(1), KnowledgeGraph(), batch_mix=type('B',(),{'episodic':1.0,'semantic':0.0,'fresh':0.0})())
        >>> scheduler.add_trace('t', np.zeros(1), 0.0)

        See Also
        --------
        next_batch
        """

        self.queue.add(
            trace_id,
            key,
            score,
            grad_overlap_proxy=grad_overlap_proxy,
            grad=grad,
            timestamp=timestamp,
        )

    def next_batch(self, batch_size: int) -> List[Tuple[str, Optional[str]]]:
        """Summary
        -------
        Return a mixed batch of replay types.

        Parameters
        ----------
        batch_size : int
            Number of items to schedule.

        Returns
        -------
        list of tuple of str and Optional[str]
            Tuples ``(kind, identifier)`` with ``kind`` in
            {``"episodic"``, ``"semantic"``, ``"fresh"``}. Episodic identifiers
            reference trace ids; others are ``None``.
        Side Effects
        ------------
        Touches the knowledge graph for statistics.

        Complexity
        ----------
        ``O(n log n)`` dominated by queue sampling.

        Examples
        --------
        >>> scheduler = ReplayScheduler(EpisodicStore(1), KnowledgeGraph(), batch_mix=type('B',(),{'episodic':1.0,'semantic':0.0,'fresh':0.0})())
        >>> scheduler.next_batch(1)[0][0]
        'episodic'

        See Also
        --------
        add_trace
        """

        n_epi = int(self.batch_mix.episodic * batch_size)
        n_sem = int(self.batch_mix.semantic * batch_size)

        epi_ids = (
            self.queue.sample(n_epi, grad_sim_threshold=self.grad_sim_threshold) if n_epi else []
        )
        # why: ensure batch size even when queue short
        epi_ids.extend([None] * (n_epi - len(epi_ids)))

        if n_sem:
            # why: touch KG for logging consistency even if result ignored
            try:
                self.kg.retrieve(np.zeros(1, dtype=float), k=1)
            except Exception:  # pragma: no cover - safety net
                pass

        batch: List[Tuple[str, Optional[str]]] = []
        batch.extend([("episodic", tid) for tid in epi_ids])
        batch.extend([("semantic", None) for _ in range(n_sem)])

        n_fresh = batch_size - n_epi - n_sem
        batch.extend([("fresh", None) for _ in range(n_fresh)])

        random.shuffle(batch)
        self._log["batches"] += 1
        return batch

    def log_status(self) -> dict:
        """Summary
        -------
        Return counters for scheduled batches.
        Returns
        -------
        dict
            Copy of internal counters.
        Examples
        --------
        >>> ReplayScheduler(EpisodicStore(1), KnowledgeGraph(), batch_mix=type('B',(),{'episodic':1.0,'semantic':0.0,'fresh':0.0})()).log_status()['batches']
        0

        See Also
        --------
        next_batch
        """

        return dict(self._log)
