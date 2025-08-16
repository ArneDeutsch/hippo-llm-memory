"""Episodic memory store backed by FAISS and SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import faiss  # type: ignore
import numpy as np


@dataclass
class TraceValue:
    """Metadata associated with a stored trace."""

    tokens_span: Optional[tuple[int, int]] = None
    entity_slots: Optional[dict] = None
    state_sketch: Optional[list] = None
    salience_tags: Optional[List[str]] = None
    provenance: Optional[str] = None


@dataclass
class DGKey:
    """Sparse k-WTA encoded key."""

    indices: np.ndarray
    values: np.ndarray
    dim: int


@dataclass
class Trace:
    """A retrieved memory trace."""

    id: int
    value: TraceValue
    key: np.ndarray
    score: float
    ts: float
    salience: float


class EpisodicStore:
    """Simple vector store for episodic memories."""

    def __init__(
        self,
        dim: int,
        db_path: str = ":memory:",
        index_str: str = "Flat",
        train_threshold: int = 100,
        *,
        config: Optional[dict] = None,
    ) -> None:
        """Create a store with given key dimensionality.

        Args:
            dim: Size of key vectors.
            db_path: Location of SQLite database for metadata.
            index_str: FAISS index factory string, e.g. ``"IVF10,PQ4"``.
            train_threshold: Number of observed keys required before training
                a quantized index.
        """

        self.dim = dim
        base = faiss.index_factory(dim, index_str, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_db()

        # Buffers for PQ training
        self._pending_keys: List[np.ndarray] = []
        self._pending_ids: List[int] = []
        self.train_threshold = train_threshold

        # Hopfield parameter (inverse temperature)
        self.beta = 1.0

        # Configuration and logging
        self.config = config or {}
        self._log = {"writes": 0, "recalls": 0, "hits": 0, "maintenance": 0}
        self._bg_thread: Optional[threading.Thread] = None
        # Maintenance bookkeeping
        self._history: List[tuple[str, object]] = []
        self._maint_log_path = self.config.get("maintenance_log")

    # ------------------------------------------------------------------
    # SQLite helpers
    def _setup_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY, value TEXT, key BLOB, ts REAL, salience REAL)"
        )
        self.conn.commit()

    def _insert_trace(self, key: np.ndarray, value_json: str, ts: float, salience: float) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO traces(value, key, ts, salience) VALUES (?, ?, ?, ?)",
            (value_json, key.tobytes(), ts, salience),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def _get_trace(self, idx: int) -> Optional[tuple[str, np.ndarray, float, float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT value, key, ts, salience FROM traces WHERE id=?", (idx,))
        row = cur.fetchone()
        if not row:
            return None
        value, key_blob, ts, salience = row
        key = np.frombuffer(key_blob, dtype="float32")
        return value, key, float(ts), float(salience)

    def keys(self) -> np.ndarray:
        """Return all stored key vectors."""

        cur = self.conn.cursor()
        cur.execute("SELECT key FROM traces")
        rows = cur.fetchall()
        if not rows:
            return np.empty((0, self.dim), dtype="float32")
        return np.vstack([np.frombuffer(r[0], dtype="float32") for r in rows])

    # ------------------------------------------------------------------
    # Encoding
    def sparse_encode(self, query: np.ndarray, k: int) -> DGKey:
        """k-Winner-Take-All sparse encoding of ``query``."""

        q = np.asarray(query, dtype="float32").reshape(-1)
        if k <= 0:
            return DGKey(
                indices=np.empty(0, dtype=np.int64), values=np.empty(0, dtype="float32"), dim=q.size
            )
        k = min(k, q.size)
        idx = np.argpartition(-np.abs(q), k - 1)[:k]
        vals = q[idx]
        return DGKey(indices=idx.astype("int64"), values=vals.astype("float32"), dim=q.size)

    # ------------------------------------------------------------------
    # Public API
    def write(self, key: np.ndarray, value: TraceValue | str) -> int:
        """Store a key/value pair."""

        if isinstance(value, str):
            value = TraceValue(provenance=value)

        key_arr = np.asarray(key, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(key_arr)
        ts = time.time()
        salience = float(len(value.salience_tags) if value.salience_tags else 1.0)
        idx = self._insert_trace(key_arr[0], json.dumps(asdict(value)), ts, salience)

        if self.index.is_trained:
            ids = np.array([idx], dtype="int64")
            self.index.add_with_ids(key_arr, ids)
        else:
            self._pending_keys.append(key_arr[0])
            self._pending_ids.append(idx)
            if len(self._pending_keys) >= self.train_threshold:
                train_mat = np.vstack(self._pending_keys)
                faiss.normalize_L2(train_mat)
                self.index.train(train_mat)
                ids = np.array(self._pending_ids, dtype="int64")
                self.index.add_with_ids(train_mat, ids)
                self._pending_keys.clear()
                self._pending_ids.clear()
        self._log["writes"] += 1
        return idx

    def recall(self, query: np.ndarray, k: int) -> List[Trace]:
        """Recall the ``k`` nearest traces for a query vector."""

        if self.index.ntotal == 0:
            return []

        query_arr = np.asarray(query, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_arr)
        scores, ids = self.index.search(query_arr, k)
        traces: List[Trace] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            trace = self._get_trace(int(idx))
            if trace is None:
                continue
            value_json, key_vec, ts, salience = trace
            value_dict = json.loads(value_json)
            value = TraceValue(**value_dict)
            traces.append(
                Trace(
                    id=int(idx),
                    value=value,
                    key=key_vec,
                    score=float(score),
                    ts=ts,
                    salience=salience,
                )
            )
        self._log["recalls"] += 1
        self._log["hits"] += len(traces)
        return traces

    def delete(self, idx: int) -> None:
        """Delete a trace by ``idx``."""

        ids = np.array([idx], dtype="int64")
        try:
            self.index.remove_ids(ids)
        except Exception:
            pass
        cur = self.conn.cursor()
        cur.execute("DELETE FROM traces WHERE id=?", (idx,))
        self.conn.commit()

    def update(
        self, idx: int, key: Optional[np.ndarray] = None, value: Optional[TraceValue] = None
    ) -> None:
        """Update the key and/or value for an existing trace."""

        cur = self.conn.cursor()
        if value is not None:
            cur.execute("UPDATE traces SET value=? WHERE id=?", (json.dumps(asdict(value)), idx))
        if key is not None:
            key_arr = np.asarray(key, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(key_arr)
            ids = np.array([idx], dtype="int64")
            try:
                self.index.remove_ids(ids)
            except Exception:
                pass
            if self.index.is_trained:
                self.index.add_with_ids(key_arr, ids)
            else:
                self._pending_keys.append(key_arr[0])
                self._pending_ids.append(idx)
            cur.execute("UPDATE traces SET key=? WHERE id=?", (key_arr[0].tobytes(), idx))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Hopfield completion
    def complete(self, query: np.ndarray, k: int = 1) -> np.ndarray:
        """Return a Hopfield-based completion of ``query`` using recalled keys."""

        traces = self.recall(query, k)
        if not traces:
            return np.asarray(query, dtype="float32")
        patterns = np.stack([t.key for t in traces])
        q = np.asarray(query, dtype="float32")
        scores = patterns @ q
        weights = np.exp(self.beta * (scores - np.max(scores)))
        weights /= np.sum(weights)
        out = weights @ patterns
        return out.astype("float32")

    # ------------------------------------------------------------------
    # Forgetting
    def decay(self, rate: float) -> None:
        """Apply exponential decay to salience values."""
        factor = max(0.0, 1.0 - rate)
        cur = self.conn.cursor()
        cur.execute("UPDATE traces SET salience = salience * ?", (factor,))
        self.conn.commit()
        self._history.append(("decay", rate))
        if self._maint_log_path:
            with open(self._maint_log_path, "a") as fh:
                fh.write(f"{time.time()} decay {rate}\n")

    def prune(self, min_salience: float = 0.1, max_age: Optional[float] = None) -> None:
        """Remove traces whose salience or age fall below thresholds."""
        cur = self.conn.cursor()
        conditions: List[str] = []
        params: List[float] = []
        if min_salience is not None:
            conditions.append("salience < ?")
            params.append(min_salience)
        if max_age is not None:
            cutoff = time.time() - max_age
            conditions.append("ts < ?")
            params.append(cutoff)
        if not conditions:
            return
        where = " OR ".join(conditions)
        cur.execute(f"SELECT id, value, key, ts, salience FROM traces WHERE {where}", params)
        rows = cur.fetchall()
        removed = []
        for idx, value_json, key_blob, ts, sal in rows:
            removed.append((idx, value_json, key_blob, ts, sal))
            self.delete(int(idx))
        if removed:
            self._history.append(("prune", removed))
            if self._maint_log_path:
                with open(self._maint_log_path, "a") as fh:
                    fh.write(f"{time.time()} prune {len(removed)}\n")

    # ------------------------------------------------------------------
    # Logging and maintenance
    def log_status(self) -> dict:
        """Return a snapshot of usage counters."""

        return dict(self._log)

    def start_background_tasks(self, interval: float = 60.0) -> None:
        """Start a thread that periodically decays and prunes memories."""

        if self._bg_thread is not None:
            return

        def loop() -> None:
            while True:
                time.sleep(interval)
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
        self._bg_thread = t

    def rollback(self, n: int = 1) -> None:
        """Undo the last ``n`` maintenance operations."""

        cur = self.conn.cursor()
        for _ in range(min(n, len(self._history))):
            op, data = self._history.pop()
            if op == "decay":
                rate = float(data)
                factor = max(0.0, 1.0 - rate)
                if factor > 0:
                    cur.execute("UPDATE traces SET salience = salience / ?", (factor,))
                    self.conn.commit()
            elif op == "prune":
                for idx, value_json, key_blob, ts, sal in data:  # type: ignore
                    cur.execute(
                        "INSERT OR REPLACE INTO traces(id, value, key, ts, salience) VALUES (?,?,?,?,?)",
                        (idx, value_json, key_blob, ts, sal),
                    )
                    key_vec = np.frombuffer(key_blob, dtype="float32")
                    faiss.normalize_L2(key_vec.reshape(1, -1))
                    if self.index.is_trained:
                        self.index.add_with_ids(
                            key_vec.reshape(1, -1), np.array([idx], dtype="int64")
                        )
                    else:
                        self._pending_keys.append(key_vec)
                        self._pending_ids.append(idx)
                self.conn.commit()
