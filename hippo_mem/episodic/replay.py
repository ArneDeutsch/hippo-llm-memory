"""Replay queue and scheduler with interference-aware batch mixing.

Summary
-------
This module implements a CA2-inspired replay system that mixes episodic,
semantic, and fresh samples.  Default batch ratios follow the 50/30/20 rule
from the evaluation plan: 50% episodic traces, 30% semantic graph recalls,
and 20% fresh items.  Items are ordered to minimise gradient interference and
the resulting batches drive downstream consolidation jobs that maintain stores
via decay, prune, and merge steps.

See Also
--------
hippo_mem.consolidation.worker.ConsolidationWorker
    Background thread that consumes batches and performs maintenance.
"""

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph


@dataclass
class ReplayItem:
    """Container for queued replay traces.

    Summary
    -------
    Each item captures metadata required for interference-aware scheduling.

    Parameters
    ----------
    trace_id:
        Identifier of the trace inside the episodic store.
    key:
        Sparse key vector with shape ``(D,)``.
    score:
        Gating score ``S`` used for prioritisation.
    step:
        Insertion step counter.
    diversity_sig:
        Average cosine dissimilarity to existing items.
    grad_overlap_proxy:
        Optional proxy for gradient overlap.
    grad:
        Optional gradient vector with shape ``(D,)``.
    timestamp:
        Time of insertion in seconds.

    Returns
    -------
    None

    Raises
    ------
    None

    Side Effects
    ------------
    None

    Complexity
    ----------
    ``O(1)`` for storage.

    Examples
    --------
    >>> ReplayItem("t1", np.zeros(4), 0.5, 1, 1.0, None, None, 0.0)
    ReplayItem(trace_id='t1', key=array([0., 0., 0., 0.], dtype=float32), score=0.5, step=1,
               diversity_sig=1.0, grad_overlap_proxy=None, grad=None, timestamp=0.0)

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
    """Queue scoring traces by salience, recency, and diversity.

    Summary
    -------
    Implements interference-aware ordering using a weighted combination of the
    salience score ``S``, temporal recency, and a diversity signature.  Optional
    gradient overlap proxies further reduce interference.

    Parameters
    ----------
    maxlen:
        Maximum number of items to keep.
    lambda1:
        Weight for salience scores.
    lambda2:
        Weight for recency.
    lambda3:
        Weight for diversity and gradient overlap.

    Returns
    -------
    None

    Raises
    ------
    None

    Side Effects
    ------------
    Internal state mutation on ``add`` and ``sample``.

    Complexity
    ----------
    ``O(n)`` for sampling where ``n`` is queue length.

    Examples
    --------
    >>> q = ReplayQueue()
    >>> q.add("t1", np.zeros(2), 1.0)
    >>> q.sample(1)
    ['t1']

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
        self.maxlen = maxlen
        self.items: List[ReplayItem] = []
        self.step = 0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def _diversity(self, key: np.ndarray, keys: np.ndarray) -> float:
        """Average cosine dissimilarity between ``key`` and ``keys``."""

        if keys.size == 0:
            return 1.0
        q = np.asarray(key, dtype="float32").reshape(1, -1)
        k = np.asarray(keys, dtype="float32")
        import faiss  # type: ignore

        faiss.normalize_L2(q)
        faiss.normalize_L2(k)
        sims = k @ q[0]
        return float(1.0 - np.mean(sims))

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
        """Add a trace to the queue.

        Summary
        -------
        Inserts a trace with associated key and gating score ``S`` while
        computing its diversity signature.

        Parameters
        ----------
        trace_id:
            Identifier of the trace.
        key:
            Key vector with shape ``(D,)``.
        score:
            Gating score ``S``.
        grad_overlap_proxy:
            Optional proxy for gradient overlap.
        grad:
            Optional gradient vector with shape ``(D,)``.
        timestamp:
            Optional insertion time in seconds; defaults to ``time.time()``.

        Returns
        -------
        None

        Raises
        ------
        None

        Side Effects
        ------------
        Mutates the internal queue state.

        Complexity
        ----------
        ``O(n)`` due to diversity computation over existing keys.

        Examples
        --------
        >>> q = ReplayQueue(maxlen=1)
        >>> q.add("t1", np.zeros(2), 0.1)
        >>> len(q.items)
        1

        See Also
        --------
        sample
            Retrieve prioritised trace identifiers.
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
        """Return cosine similarity between vectors."""

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _priority_scores(self) -> List[float]:
        """Return priority scores for the current queue items."""

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
        """Select indices respecting gradient overlap constraints."""

        selected: List[int] = []
        last_grad: Optional[np.ndarray] = None
        for idx in order:
            if len(selected) >= k:
                break
            item = self.items[idx]
            if last_grad is not None and item.grad is not None:
                if self._cosine(last_grad, item.grad) >= grad_sim_threshold:
                    continue
            selected.append(idx)
            if item.grad is not None:
                last_grad = item.grad
        return selected

    def sample(self, k: int, *, grad_sim_threshold: float = 0.9) -> List[str]:
        """Retrieve trace identifiers ordered by priority.

        Summary
        -------
        Picks up to ``k`` traces respecting a cosine similarity threshold on
        gradient vectors to minimise interference.

        Parameters
        ----------
        k:
            Maximum number of trace identifiers to return.
        grad_sim_threshold:
            Maximum allowed cosine similarity between successive gradients.

        Returns
        -------
        List[str]
            Selected trace identifiers.

        Raises
        ------
        None

        Side Effects
        ------------
        Accesses but does not modify the queue state.

        Complexity
        ----------
        ``O(n log n)`` for sorting ``n`` items.

        Examples
        --------
        >>> q = ReplayQueue()
        >>> q.add("t1", np.zeros(2), 1.0)
        >>> q.sample(1)
        ['t1']

        See Also
        --------
        add
            Insert traces into the queue.
        """

        if not self.items:
            return []
        priorities = self._priority_scores()
        order = list(np.argsort(priorities)[::-1])
        selected = self._select_indices(order, k, grad_sim_threshold)
        return [self.items[i].trace_id for i in selected]


class BatchMixLike(Protocol):
    """Protocol describing the batch mix structure."""

    episodic: float
    semantic: float
    fresh: float


class ReplayScheduler:
    """Produce replay batches mixing episodic, semantic, and fresh items.

    Summary
    -------
    Coordinates replay according to a batch mix, typically ``(0.5, 0.3, 0.2)``
    for episodic, semantic, and fresh samples.  Scheduling respects gradient
    similarity thresholds to mitigate interference.

    Parameters
    ----------
    store:
        EpisodicStore holding trace payloads.
    kg:
        Knowledge graph store for semantic recalls.
    batch_mix:
        Object with ``episodic``, ``semantic``, and ``fresh`` fields indicating
        target ratios.
    grad_sim_threshold:
        Maximum cosine similarity between successive gradients.

    Returns
    -------
    None

    Raises
    ------
    None

    Side Effects
    ------------
    Maintains internal counters for diagnostics.

    Complexity
    ----------
    ``O(n log n)`` per batch where ``n`` is queue length.

    Examples
    --------
    >>> mix = type("Mix", (), {"episodic": 0.5, "semantic": 0.3, "fresh": 0.2})()
    >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
    >>> sched.log_status()["batches"]
    0

    See Also
    --------
    ReplayQueue
        Underlying priority queue.
    """

    def __init__(
        self,
        store: EpisodicStore,
        kg: KnowledgeGraph,
        *,
        batch_mix: BatchMixLike,
        grad_sim_threshold: float = 0.9,
    ) -> None:
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
        """Enqueue a trace for future replay.

        Summary
        -------
        Forwards to :meth:`ReplayQueue.add` and records nothing on failure.

        Parameters
        ----------
        trace_id:
            Identifier of the trace in ``store``.
        key:
            Key vector with shape ``(D,)``.
        score:
            Gating score ``S``.
        grad_overlap_proxy:
            Optional proxy for gradient overlap.
        grad:
            Optional gradient vector with shape ``(D,)``.
        timestamp:
            Optional insertion time in seconds.

        Returns
        -------
        None

        Raises
        ------
        None

        Side Effects
        ------------
        Mutates the internal queue state.

        Complexity
        ----------
        ``O(n)`` due to diversity computation.

        Examples
        --------
        >>> mix = type("Mix", (), {"episodic": 1.0, "semantic": 0.0, "fresh": 0.0})()
        >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
        >>> sched.add_trace("t1", np.zeros(2), 0.2)
        >>> sched.queue.items[0].trace_id
        't1'

        See Also
        --------
        ReplayQueue.add
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
        """Return a mixed batch of replay types.

        Summary
        -------
        Samples episodic identifiers from the queue and composes a batch
        including semantic and fresh placeholders according to ``batch_mix``.

        Parameters
        ----------
        batch_size:
            Number of items to schedule.

        Returns
        -------
        list[tuple[str, Optional[str]]]
            Tuples of ``(kind, identifier)`` where ``kind`` is one of
            ``"episodic"``, ``"semantic"`` or ``"fresh"``.  Episodic identifiers
            refer to trace ids; others are ``None``.

        Raises
        ------
        None

        Side Effects
        ------------
        Increments internal batch counters and performs dummy KG retrieval for
        logging.

        Complexity
        ----------
        ``O(n log n)`` for queue sampling.

        Examples
        --------
        >>> mix = type("Mix", (), {"episodic": 0.5, "semantic": 0.3, "fresh": 0.2})()
        >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
        >>> sched.next_batch(2)
        [('fresh', None), ('episodic', None)]

        See Also
        --------
        log_status
            Inspect counters.
        """

        n_epi = int(self.batch_mix.episodic * batch_size)
        n_sem = int(self.batch_mix.semantic * batch_size)

        epi_ids = (
            self.queue.sample(n_epi, grad_sim_threshold=self.grad_sim_threshold) if n_epi else []
        )
        # If we requested more episodic items than available fill with ``None``
        epi_ids.extend([None] * (n_epi - len(epi_ids)))

        if n_sem:
            # Touch the KG to keep statistics/logs consistent; the actual
            # returned graph is ignored by the training loop in this minimal
            # implementation.
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
        """Return counters for scheduled batches.

        Summary
        -------
        Reports how many batches have been produced.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with diagnostic counters.

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
        >>> mix = type("Mix", (), {"episodic": 1.0, "semantic": 0.0, "fresh": 0.0})()
        >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
        >>> sched.log_status()
        {'batches': 0}

        See Also
        --------
        next_batch
        """

        return dict(self._log)
