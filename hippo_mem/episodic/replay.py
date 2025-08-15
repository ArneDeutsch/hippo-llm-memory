"""Replay queue utilities for episodic memory."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph


@dataclass
class ReplayItem:
    """Metadata stored for replay scheduling."""

    trace_id: str
    key: np.ndarray
    score: float
    step: int
    diversity_sig: float
    grad_overlap_proxy: Optional[float]
    timestamp: float


class ReplayQueue:
    """Replay queue mixing gating score, recency and diversity.

    Items are ordered by a weighted combination of gating score ``S``, recency
    and a diversity signature augmented by an optional gradient-overlap proxy.
    ``lambda1``, ``lambda2`` and ``lambda3`` control the relative weighting of
    these signals.
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
        timestamp: float | None = None,
    ) -> None:
        """Add a trace with associated key and gating score ``S``."""

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
            timestamp=timestamp,
        )
        self.items.append(item)
        if len(self.items) > self.maxlen:
            self.items.pop(0)

    def sample(self, k: int) -> List[str]:
        """Return ``k`` trace identifiers in priority order."""

        if not self.items:
            return []
        now = self.step
        priorities = []
        for it in self.items:
            recency = 1.0 / (now - it.step + 1)
            diversity_component = it.diversity_sig
            if it.grad_overlap_proxy is not None:
                diversity_component += it.grad_overlap_proxy
            p = (
                self.lambda1 * it.score
                + self.lambda2 * recency
                + self.lambda3 * diversity_component
            )
            priorities.append(p)
        order = np.argsort(priorities)[::-1]
        return [self.items[i].trace_id for i in order[:k]]


class BatchMixLike(Protocol):
    """Protocol describing the batch mix structure."""

    episodic: float
    semantic: float
    fresh: float


class ReplayScheduler:
    """Scheduler that interleaves episodic, semantic and fresh items.

    The scheduler maintains an internal :class:`ReplayQueue` for episodic
    traces and produces batches mixing episodic replays, semantic recalls from
    the knowledge graph and fresh data according to provided ratios.
    """

    def __init__(
        self,
        store: EpisodicStore,
        kg: KnowledgeGraph,
        *,
        batch_mix: BatchMixLike,
    ) -> None:
        self.store = store
        self.kg = kg
        self.batch_mix = batch_mix
        self.queue = ReplayQueue()
        self._log = {"batches": 0}

    def add_trace(
        self,
        trace_id: str,
        key: np.ndarray,
        score: float,
        *,
        grad_overlap_proxy: Optional[float] = None,
        timestamp: float | None = None,
    ) -> None:
        """Proxy for :meth:`ReplayQueue.add` to enqueue traces."""

        self.queue.add(
            trace_id,
            key,
            score,
            grad_overlap_proxy=grad_overlap_proxy,
            timestamp=timestamp,
        )

    def next_batch(self, batch_size: int) -> List[Tuple[str, Optional[str]]]:
        """Return a mixed batch of replay types.

        Args:
            batch_size: Number of items to schedule.

        Returns:
            A list of ``(kind, identifier)`` tuples where ``kind`` is one of
            ``"episodic"``, ``"semantic"`` or ``"fresh"``.  For episodic
            items the identifier corresponds to a trace id selected by the
            internal queue.  Identifiers for other kinds are ``None``.
        """

        n_epi = int(self.batch_mix.episodic * batch_size)
        n_sem = int(self.batch_mix.semantic * batch_size)

        epi_ids = self.queue.sample(n_epi) if n_epi else []
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
        """Return counters for scheduled batches."""

        return dict(self._log)
