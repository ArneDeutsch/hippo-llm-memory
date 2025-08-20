"""Algorithm Card: HEI-NW AssocStore

Summary
-------
Persistent store of sparse episodic traces backing the HEI-NW adapter.

Integration style
-----------------
Cross-attention adapter queries recalled traces.

Data structures
---------------
``DGKey`` sparse keys, ``TraceValue`` metadata, ``AssocStore`` FAISS index,
``ReplayQueue`` for consolidation.

Pipeline
--------
1. WriteGate computes ``S = α·surprise + β·novelty + γ·reward + δ·pin``.
2. If ``S > τ`` persist ``(DGKey, TraceValue)`` to the store.
3. Recall uses FAISS KNN followed by Hopfield completion.
4. Enqueue trace id into ``ReplayQueue``.

Design rationale & trade-offs
-----------------------------
Product-quantized indices reduce RAM while Hopfield completion recovers full
patterns.  Trade-off: extra compute during recall.

Failure modes & diagnostics
---------------------------
Missing recalls → ensure index trained; DB mismatch → inspect ``log_status``.

Ablation switches & expected effects
------------------------------------
``use_sparsity=false`` stores dense keys causing interference.  ``use_completion=false``
skips Hopfield, lowering recall F1.

Contracts
---------
Writes persist to SQLite and FAISS; duplicate writes with same id are idempotent.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Union

import numpy as np

from hippo_mem._faiss import faiss

from .db import TraceDB
from .gating import DGKey, densify, k_wta
from .index import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class TraceValue:
    """Metadata associated with a stored trace.

    Summary
    -------
    Optional fields describing an episode.

    Parameters
    ----------
    tokens_span : tuple of int, optional
        Token indices ``(start, end)``.
    entity_slots : dict, optional
        Mapping of entities ``{role: token}``.
    state_sketch : list, optional
        Abstract state representation.
    salience_tags : list of str, optional
        Tags contributing to initial salience.
    provenance : str, optional
        Source identifier.
    Examples
    --------
    >>> TraceValue(provenance="unit").provenance
    'unit'

    See Also
    --------
    Trace
    """

    tokens_span: Optional[tuple[int, int]] = None
    entity_slots: Optional[dict] = None
    state_sketch: Optional[list] = None
    salience_tags: Optional[List[str]] = None
    provenance: Optional[str] = None


@dataclass
class Trace:
    """A retrieved memory trace.

    Summary
    -------
    Container returned by :meth:`EpisodicStore.recall`.

    Parameters
    ----------
    id : int
        Unique identifier.
    value : TraceValue
        Payload metadata.
    key : DGKey
        Sparse key representing the episode.
    score : float
        Similarity score from FAISS.
    ts : float
        Unix timestamp.
    salience : float
        Current salience value.
    Examples
    --------
    >>> Trace(1, TraceValue(), DGKey(np.array([]), np.array([], dtype=np.float32), 0), 0.0, 0.0, 0.0).id
    1

    See Also
    --------
    EpisodicStore.recall
    """

    id: int
    value: TraceValue
    key: DGKey
    score: float
    ts: float
    salience: float


class EpisodicStore:
    """Simple vector store for episodic memories.

    Summary
    -------
    Stores sparse keys and metadata with optional Hopfield completion.
    """

    def __init__(
        self,
        dim: int,
        db_path: str = ":memory:",
        index_str: str = "Flat",
        train_threshold: int = 100,
        *,
        k_wta: int = 0,
        config: Optional[dict] = None,
    ) -> None:
        """Create a store with given key dimensionality.

        Parameters
        ----------
        dim : int
            Size of key vectors ``d``.
        db_path : str, optional
            Location of SQLite database for metadata.
        index_str : str, optional
            FAISS index factory string, e.g. ``"IVF10,PQ4"``.
        train_threshold : int, optional
            Number of observed keys required before training a quantized index.
        k_wta : int, optional
            Number of winners for sparse encoding; ``0`` keeps dense keys.
        config : dict, optional
            Additional configuration options.
        Side Effects
        ------------
        Opens SQLite connection and builds FAISS index.
        Examples
        --------
        >>> store = EpisodicStore(4)
        >>> store.dim
        4

        See Also
        --------
        write, recall
        """

        self.dim = dim
        self.k_wta = k_wta
        self.index = VectorIndex(dim, index_str, train_threshold)
        self.db = TraceDB(db_path)

        # Hopfield parameter (inverse temperature)
        self.beta = 1.0

        # Configuration and logging
        self.config = config or {}
        self._log = {"writes": 0, "recalls": 0, "hits": 0, "requests": 0, "maintenance": 0}
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._history: List[dict[str, Any]] = []
        self._max_undo = int(self.config.get("max_undo", 5))
        self._maintenance_log: List[dict[str, Any]] = []
        self._log_file = self.config.get("maintenance_log")

    # ------------------------------------------------------------------
    def keys(self) -> np.ndarray:
        """Return all stored key vectors.

        Summary
        -------
        Retrieve dense representations of every stored key.
        Returns
        -------
        numpy.ndarray
        Array with shape ``(n, d)``.
        Complexity
        ----------
        ``O(n d)`` to load from the database.

        Examples
        --------
        >>> EpisodicStore(2).keys().shape
        (0, 2)

        See Also
        --------
        write
        """

        return self.db.keys(self.dim)

    # ------------------------------------------------------------------
    # Maintenance helpers
    def _log_event(self, op: str, info: dict[str, Any]) -> None:
        event = {"ts": time.time(), "op": op, **info}
        self._maintenance_log.append(event)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")

    def _push_history(self, op: str, rows: list[Any]) -> None:
        self._history.append({"op": op, "rows": rows})
        if len(self._history) > self._max_undo:
            self._history.pop(0)

    # ------------------------------------------------------------------
    # Encoding
    def sparse_encode(self, query: np.ndarray, k: int) -> DGKey:
        """k-Winner-Take-All sparse encoding of ``query``.

        Summary
        -------
        Convenience wrapper around :func:`k_wta`.

        Parameters
        ----------
        query : numpy.ndarray
            Dense vector of shape ``(d,)``.
        k : int
            Number of winners.

        Returns
        -------
        DGKey
            Sparse key representing ``query``.
        Complexity
        ----------
        ``O(d)``.

        Examples
        --------
        >>> EpisodicStore(3).sparse_encode(np.ones(3, dtype=np.float32), 2).indices.size
        2

        See Also
        --------
        k_wta
        """

        return k_wta(query, k)

    def _to_dense(self, key: DGKey) -> np.ndarray:
        """Convert ``key`` back to a dense vector.

        Summary
        -------
        Internal helper to expand sparse keys.

        Parameters
        ----------
        key : DGKey
            Sparse key.

        Returns
        -------
        numpy.ndarray
            Dense vector ``(d,)``.
        Complexity
        ----------
        ``O(d)``.

        Examples
        --------
        >>> store = EpisodicStore(2)
        >>> store._to_dense(DGKey(np.array([0]), np.array([1.], dtype=np.float32), 2))
        array([1., 0.], dtype=float32)

        See Also
        --------
        densify
        """

        return densify(key)

    # ------------------------------------------------------------------
    # Public API
    def write(self, key: Union[np.ndarray, DGKey], value: TraceValue | str) -> int:
        """Store a key/value pair.

        Summary
        -------
        Persist an episode and return its identifier.

        Parameters
        ----------
        key : numpy.ndarray or DGKey
            Episode key of shape ``(d,)`` or sparse variant.
        value : TraceValue or str
            Metadata or provenance string.

        Returns
        -------
        int
            Assigned trace identifier.
        Side Effects
        ------------
        Writes to SQLite and FAISS indices.

        Complexity
        ----------
        ``O(d)`` for encoding plus index insertion.

        Examples
        --------
        >>> store = EpisodicStore(2)
        >>> store.write(np.ones(2, dtype=np.float32), "unit") >= 0
        True

        See Also
        --------
        recall
        """

        if isinstance(value, str):
            value = TraceValue(provenance=value)

        if isinstance(key, np.ndarray):
            if self.k_wta > 0:
                key = self.sparse_encode(key, self.k_wta)
            else:
                idxs = np.nonzero(key)[0]
                key = DGKey(
                    indices=idxs.astype("int64"), values=key[idxs].astype("float32"), dim=key.size
                )

        key_arr = self._to_dense(key).reshape(1, -1)
        faiss.normalize_L2(key_arr)
        ts = time.time()
        # why: tags approximate initial salience when explicit score absent
        salience = float(len(value.salience_tags) if value.salience_tags else 1.0)
        idx = self.db.insert(key_arr[0], json.dumps(asdict(value)), ts, salience)
        self.index.add(key_arr, idx)
        self._log["writes"] += 1
        return idx

    def recall(self, query: np.ndarray, k: int) -> List[Trace]:
        """Recall the ``k`` nearest traces for a query vector.

        Summary
        -------
        Performs cosine search and reconstructs ``Trace`` records.

        Parameters
        ----------
        query : numpy.ndarray
            Query vector of shape ``(d,)``.
        k : int
            Number of neighbours.

        Returns
        -------
        list of Trace
            Retrieved traces sorted by similarity.
        Side Effects
        ------------
        Increments internal counters.

        Complexity
        ----------
        ``O(k d)`` after FAISS search.

        Examples
        --------
        >>> store = EpisodicStore(2)
        >>> _ = store.write(np.ones(2, dtype=np.float32), "unit")
        >>> len(store.recall(np.ones(2, dtype=np.float32), 1))
        1

        See Also
        --------
        write
        """

        if self.index.ntotal == 0:
            return []

        query_arr = np.asarray(query, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_arr)
        scores, ids = self.index.search(query_arr, k)
        traces: List[Trace] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                # why: FAISS pads results with -1 when fewer than k matches
                continue
            trace = self.db.get(int(idx))
            if trace is None:
                continue
            value_json, key_vec, ts, salience = trace
            value_dict = json.loads(value_json)
            value = TraceValue(**value_dict)
            dg_key = DGKey(
                indices=np.nonzero(key_vec)[0].astype("int64"),
                values=key_vec[np.nonzero(key_vec)].astype("float32"),
                dim=self.dim,
            )
            traces.append(
                Trace(
                    id=int(idx),
                    value=value,
                    key=dg_key,
                    score=float(score),
                    ts=ts,
                    salience=salience,
                )
            )
        self._log["recalls"] += 1
        self._log["hits"] += len(traces)
        self._log["requests"] += k
        return traces

    def to_dense(self, key: DGKey) -> np.ndarray:
        """Convert a sparse ``DGKey`` into a dense vector."""

        return self._to_dense(key)

    def delete(self, idx: int) -> None:
        """Delete a trace by ``idx``.

        Summary
        -------
        Removes the entry from both FAISS and SQLite.

        Parameters
        ----------
        idx : int
            Trace identifier.
        Side Effects
        ------------
        Modifies persistent storage.
        Examples
        --------
        >>> store = EpisodicStore(2)
        >>> tid = store.write(np.ones(2, dtype=np.float32), "unit")
        >>> store.delete(tid)
        >>> store.recall(np.ones(2, dtype=np.float32), 1)
        []

        See Also
        --------
        write
        """

        self.index.remove(idx)
        self.db.delete(idx)

    def update(
        self,
        idx: int,
        key: Optional[Union[np.ndarray, DGKey]] = None,
        value: Optional[TraceValue] = None,
    ) -> None:
        """Update the key and/or value for an existing trace.

        Summary
        -------
        Overwrites stored key/value for a given identifier.

        Parameters
        ----------
        idx : int
            Trace identifier.
        key : numpy.ndarray or DGKey, optional
            New key.
        value : TraceValue, optional
            New metadata.
        Side Effects
        ------------
        Mutates persistent storage.

        Complexity
        ----------
        ``O(d)`` for key updates.

        Examples
        --------
        >>> store = EpisodicStore(2)
        >>> tid = store.write(np.ones(2, dtype=np.float32), "unit")
        >>> store.update(tid, value=TraceValue(provenance="x"))

        See Also
        --------
        write
        """

        if value is not None:
            self.db.update_value(idx, json.dumps(asdict(value)))
        if key is not None:
            if isinstance(key, np.ndarray):
                if self.k_wta > 0:
                    key = self.sparse_encode(key, self.k_wta)
                else:
                    idxs = np.nonzero(key)[0]
                    key = DGKey(
                        indices=idxs.astype("int64"),
                        values=key[idxs].astype("float32"),
                        dim=key.size,
                    )
            key_arr = self._to_dense(key).reshape(1, -1)
            faiss.normalize_L2(key_arr)
            self.index.update(key_arr, idx)
            self.db.update_key(idx, key_arr[0])

    # ------------------------------------------------------------------
    # Hopfield completion
    def complete(self, query: np.ndarray, k: int = 1) -> np.ndarray:
        """Return a Hopfield-based completion of ``query`` using recalled keys.

        Summary
        -------
        Modern-Hopfield readout to densify sparse cues.

        Parameters
        ----------
        query : numpy.ndarray
            Input cue of shape ``(d,)``.
        k : int, optional
            Number of patterns to recall.

        Returns
        -------
        numpy.ndarray
            Completed vector of shape ``(d,)``.
        Complexity
        ----------
        ``O(k d)``.

        Examples
        --------
        >>> store = EpisodicStore(2, config={"hopfield": False})
        >>> store.complete(np.ones(2, dtype=np.float32)).shape
        (2,)

        See Also
        --------
        recall
        """

        if not self.config.get("hopfield", True):
            return np.asarray(query, dtype="float32")

        traces = self.recall(query, k)
        if not traces:
            return np.asarray(query, dtype="float32")
        patterns = np.stack([self._to_dense(t.key) for t in traces])
        q = np.asarray(query, dtype="float32")
        scores = patterns @ q
        # why: softmax weights emphasise closer patterns
        weights = np.exp(self.beta * (scores - np.max(scores)))
        weights /= np.sum(weights)
        out = weights @ patterns
        return out.astype("float32")

    # ------------------------------------------------------------------
    # Forgetting
    def decay(self, rate: float) -> None:
        """Apply exponential decay to salience values.

        Summary
        -------
        Multiplies each salience by ``1 - rate``.

        Parameters
        ----------
        rate : float
            Decay rate ``0 ≤ rate ≤ 1``.
        Side Effects
        ------------
        Persists updated salience and logs operation.

        Complexity
        ----------
        ``O(n)`` over stored traces.

        Examples
        --------
        >>> store = EpisodicStore(2)
        >>> store.decay(0.1)

        See Also
        --------
        prune
        """

        factor = max(0.0, 1.0 - rate)
        rows = self.db.decay(factor)
        if rows:
            self._push_history("decay", rows)
        self._log_event("decay", {"rate": rate})

    def prune(self, min_salience: float = 0.1, max_age: Optional[float] = None) -> None:
        """Remove traces whose salience or age fall below thresholds.

        Summary
        -------
        Deletes stale or insignificant memories.

        Parameters
        ----------
        min_salience : float, optional
            Minimum allowed salience.
        max_age : float, optional
            Maximum age in seconds.
        Side Effects
        ------------
        Writes deletions to persistent storage and logs operation.

        Complexity
        ----------
        ``O(n)`` to scan candidates.

        Examples
        --------
        >>> store = EpisodicStore(2)
        >>> store.prune()

        See Also
        --------
        decay
        """

        cutoff = time.time() - max_age if max_age is not None else None
        rows = self.db.fetch_prune_candidates(min_salience, cutoff)
        if not rows:
            return
        self._push_history("prune", rows)
        for row in rows:
            self.delete(int(row[0]))
        self._log_event("prune", {"min_salience": min_salience, "max_age": max_age})

    # ------------------------------------------------------------------
    # Logging and maintenance
    def log_status(self) -> dict:
        """Return a snapshot of usage counters.

        Summary
        -------
        Provides diagnostic counters since instantiation.
        Returns
        -------
        dict
            Copy of internal counters.
        Examples
        --------
        >>> EpisodicStore(1).log_status()["writes"]
        0

        See Also
        --------
        start_background_tasks
        """

        log = dict(self._log)
        req = log.get("requests", 0)
        log["hit_rate"] = log["hits"] / req if req else 0.0
        return log

    def start_background_tasks(self, interval: float = 60.0) -> None:
        """Start a thread that periodically decays and prunes memories.

        Summary
        -------
        Spawns a daemon thread for maintenance.

        Parameters
        ----------
        interval : float, optional
            Sleep time between maintenance runs in seconds.
        Side Effects
        ------------
        Launches a background thread.
        Examples
        --------
        >>> store = EpisodicStore(1)
        >>> store.start_background_tasks(0.01)

        See Also
        --------
        decay, prune
        """

        if self._bg_thread is not None:
            return

        stop_event = threading.Event()

        def loop() -> None:
            # why: maintain store health without blocking caller
            while not stop_event.wait(interval):
                rate = float(self.config.get("decay_rate", 0.0))
                if rate > 0:
                    self.decay(rate)
                prune_cfg = self.config.get("prune", {})
                self.prune(
                    float(prune_cfg.get("min_salience", 0.1)),
                    prune_cfg.get("max_age"),
                )
                self._log["maintenance"] += 1

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._stop_event = stop_event
        self._bg_thread = t

    def stop_background_tasks(self) -> None:
        """Stop background maintenance thread if running.

        Summary
        -------
        Idempotently signals the maintenance loop to exit and waits
        briefly for the thread to terminate.
        """

        if self._bg_thread is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        self._bg_thread.join(timeout=1.0)
        self._bg_thread = None
        self._stop_event = None

    def rollback(self, n: int = 1) -> None:
        """Rollback the last ``n`` maintenance operations.

        Summary
        -------
        Restores deleted/decayed entries from history.

        Parameters
        ----------
        n : int, optional
            Number of steps to undo.
        Side Effects
        ------------
        Modifies store contents and logs event.

        Complexity
        ----------
        ``O(n d)`` for key restoration.

        Examples
        --------
        >>> store = EpisodicStore(1)
        >>> store.rollback(0)

        See Also
        --------
        decay, prune
        """

        for _ in range(n):
            if not self._history:
                break
            entry = self._history.pop()
            op = entry["op"]
            rows = entry["rows"]
            if op == "decay":
                self.db.restore_salience(rows)
            elif op == "prune":
                self.db.restore_rows(rows)
                for idx, value_json, key_blob, ts, salience in rows:
                    # why: reconstruct dense vector before re-adding to FAISS
                    key_vec = np.frombuffer(key_blob, dtype="float32").reshape(1, -1)
                    faiss.normalize_L2(key_vec)
                    self.index.add(key_vec, int(idx))
            self._log_event("rollback", {"op": op})


class AsyncStoreWriter:
    """Background worker that commits writes asynchronously.

    Summary
    -------
    Uses a thread and queue to decouple writes from callers.

    Parameters
    ----------
    store : EpisodicStore
        Target store receiving committed writes.
    maxsize : int, optional
        Queue capacity.

    Examples
    --------
    >>> store = EpisodicStore(2)
    >>> writer = AsyncStoreWriter(store)
    >>> writer.stop()
    """

    def __init__(self, store: EpisodicStore, maxsize: int = 64) -> None:
        self.store = store
        self.queue: queue.Queue[tuple[Union[np.ndarray, DGKey], TraceValue]] = queue.Queue(maxsize)
        self.stats = {"writes_enqueued": 0, "writes_committed": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, key: Union[np.ndarray, DGKey], value: TraceValue | str) -> None:
        if isinstance(value, str):
            value = TraceValue(provenance=value)
        self.stats["writes_enqueued"] += 1
        self.queue.put((key, value))

    def _worker(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                key, value = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.store.write(key, value)
                self.stats["writes_committed"] += 1
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


__all__ = ["TraceValue", "Trace", "EpisodicStore", "AsyncStoreWriter"]
