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
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

import hippo_mem.common.io as io
from hippo_mem._faiss import faiss
from hippo_mem.common.history import HistoryEntry, RollbackMixin
from hippo_mem.common.lifecycle import StoreLifecycleMixin

from .event_logger import EventLogger
from .gating import DGKey, densify, k_wta
from .index import FaissIndex, IndexStrategy
from .maintenance import Decayer, Pruner
from .persistence import TracePersistence
from .types import Trace, TraceValue

logger = logging.getLogger(__name__)


class EpisodicStore(StoreLifecycleMixin, RollbackMixin):
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
        index: IndexStrategy | None = None,
        persistence: TracePersistence | None = None,
        k_wta: int = 0,
        config: Optional[dict] = None,
        pruner: Optional[Pruner] = None,
        decayer: Optional[Decayer] = None,
        logger: Optional[EventLogger] = None,
        key_noise: float = 0.0,
        seed: int = 0,
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
        self.index = index or FaissIndex(dim, index_str, train_threshold)
        self.persistence = persistence or TracePersistence(db_path)
        self.key_noise = float(key_noise)
        self._rng = np.random.default_rng(seed)

        # Hopfield parameter (inverse temperature)
        self.beta = 1.0

        # Configuration and helpers
        self.config = config or {}
        self.logger = logger or EventLogger(self.config.get("maintenance_log"))
        self._maintenance_log = self.logger._events
        self.pruner = pruner or Pruner()
        self.decayer = decayer or Decayer()
        StoreLifecycleMixin.__init__(self)
        RollbackMixin.__init__(self, int(self.config.get("max_undo", 5)))

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

        return self.persistence.keys(self.dim)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Encoding
    def sparse_encode(self, query: np.ndarray, k: int) -> DGKey:
        """k-Winner-Take-All sparse encoding of ``query``.

        Summary
        -------
        Convenience wrapper around :func:`k_wta` for both writes and cues.

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
            if self.key_noise > 0:
                noise = self._rng.normal(0.0, self.key_noise, size=key.shape).astype("float32")
                key = key + noise
            if self.k_wta > 0:
                key = self.sparse_encode(key, self.k_wta)
            else:
                idxs = np.nonzero(key)[0]
                key = DGKey(
                    indices=idxs.astype("int64"), values=key[idxs].astype("float32"), dim=key.size
                )

        if value.tokens_span is None:
            value.tokens_span = (0, 0)
        if value.entity_slots is None:
            value.entity_slots = {"provenance": value.provenance}
        if value.state_sketch is None:
            value.state_sketch = [value.provenance]
        if not value.salience_tags:
            value.salience_tags = [value.provenance or "auto"]

        key_arr = self._to_dense(key).reshape(1, -1)
        faiss.normalize_L2(key_arr)
        ts = time.time()
        # why: tags approximate initial salience when explicit score absent
        salience = float(len(value.salience_tags) if value.salience_tags else 1.0)
        idx = self.persistence.insert(key_arr[0], value, ts, salience)
        self.index.add(key_arr, idx)
        self.logger.increment("writes")
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
            trace = self.persistence.get(int(idx))
            if trace is None:
                continue
            value, key_vec, ts, salience = trace
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
        self.logger.increment("recalls")
        self.logger.increment("hits", len(traces))
        self.logger.increment("requests", k)
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
        self.persistence.delete(idx)

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
            self.persistence.update_value(idx, value)
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
            self.persistence.update_key(idx, key_arr[0])

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

        rows = self.decayer.decay(self, rate)
        if rows:
            self._push_history("decay", rows)
        self.logger.log_event("decay", {"rate": rate})

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

        rows = self.pruner.prune(self, min_salience, max_age)
        if not rows:
            return
        self._push_history("prune", rows)
        self.logger.log_event("prune", {"min_salience": min_salience, "max_age": max_age})

    # ------------------------------------------------------------------
    # Persistence
    def save(
        self,
        directory: str,
        session_id: str,
        fmt: str = "jsonl",
        replay_samples: int = 0,
        gate_attempts: int = 0,
    ) -> None:
        """Save all traces under ``directory/session_id``.

        Parameters
        ----------
        directory : str
            Output directory.
        session_id : str
            Subdirectory name for this session.
        fmt : str, optional
            ``"jsonl"`` (default) or ``"parquet"``.
        replay_samples : int, optional
            Number of replayed samples driving persistence.
        gate_attempts : int, optional
            Gate attempts during teach; marks source as ``"teach"`` when positive.
        """

        path = Path(directory) / session_id
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "schema": "episodic.store_meta.v1",
            "replay_samples": int(replay_samples),
            "source": (
                "replay" if replay_samples > 0 else "teach" if gate_attempts > 0 else "stub"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        io.atomic_write_json(path / "store_meta.json", meta)
        if replay_samples <= 0:
            return

        records = []
        for idx, value, key_vec, ts, salience in self.persistence.all():
            records.append(
                {
                    "schema": "episodic.v1",
                    "id": idx,
                    "key": key_vec.tolist(),
                    "value": asdict(value),
                    "ts": ts,
                    "salience": salience,
                }
            )
        file = path / f"episodic.{fmt}"
        if fmt == "jsonl":
            io.atomic_write_jsonl(file, records)
        elif fmt == "parquet":
            try:
                import pandas as pd
            except Exception as exc:  # pragma: no cover - optional
                raise RuntimeError("Parquet support requires pandas") from exc
            io.atomic_write_file(
                file, lambda tmp: pd.DataFrame(records).to_parquet(tmp, index=False)
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {fmt}")

    def load(self, directory: str, session_id: str, fmt: str = "jsonl") -> None:
        """Load traces from ``directory/session_id``.

        Existing content is replaced; use on a new store instance.

        Parameters
        ----------
        directory : str
            Input directory.
        session_id : str
            Session identifier.
        fmt : str, optional
            ``"jsonl"`` (default) or ``"parquet"``.
        """

        path = Path(directory) / session_id / f"episodic.{fmt}"
        rows: list[tuple[int, str, bytes, float, float]] = []
        if fmt == "jsonl":
            for rec in io.read_jsonl(path):
                key_arr = np.asarray(rec["key"], dtype="float32")
                rows.append(
                    (
                        int(rec["id"]),
                        json.dumps(rec["value"]),
                        key_arr.tobytes(),
                        float(rec["ts"]),
                        float(rec["salience"]),
                    )
                )
                key_vec = key_arr.reshape(1, -1)
                faiss.normalize_L2(key_vec)
                self.index.add(key_vec, int(rec["id"]))
        elif fmt == "parquet":
            df = io.read_parquet(path)
            for rec in df.to_dict(orient="records"):
                key_arr = np.asarray(rec["key"], dtype="float32")
                rows.append(
                    (
                        int(rec["id"]),
                        json.dumps(rec["value"]),
                        key_arr.tobytes(),
                        float(rec["ts"]),
                        float(rec["salience"]),
                    )
                )
                key_vec = key_arr.reshape(1, -1)
                faiss.normalize_L2(key_vec)
                self.index.add(key_vec, int(rec["id"]))
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {fmt}")
        self.persistence.restore_rows(rows)

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

        return self.logger.status()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    def _maintenance_tick(self, _event: threading.Event) -> None:
        rate = float(self.config.get("decay_rate", 0.0))
        if rate > 0:
            self.decay(rate)
        prune_cfg = self.config.get("prune", {})
        self.prune(
            float(prune_cfg.get("min_salience", 0.1)),
            prune_cfg.get("max_age"),
        )
        self.logger.increment("maintenance")

    def _apply_rollback(self, entry: HistoryEntry) -> None:
        op = entry.op
        rows = entry.data
        if op == "decay":
            self.persistence.restore_salience(rows)
        elif op == "prune":
            self.persistence.restore_rows(rows)
            for idx, _value_json, key_blob, _ts, _salience in rows:
                key_vec = np.frombuffer(key_blob, dtype="float32").reshape(1, -1)
                faiss.normalize_L2(key_vec)
                self.index.add(key_vec, int(idx))
        self.logger.log_event("rollback", {"op": op})


__all__ = ["TraceValue", "Trace", "EpisodicStore"]
