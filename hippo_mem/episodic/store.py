"""Episodic memory store backed by FAISS and SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import faiss  # type: ignore
import numpy as np


@dataclass
class Trace:
    """A retrieved memory trace."""

    id: int
    value: str
    key: np.ndarray
    score: float


class EpisodicStore:
    """Simple vector store for episodic memories."""

    def __init__(self, dim: int, db_path: str = ":memory:") -> None:
        """Create a store with given key dimensionality.

        Args:
            dim: Size of key vectors.
            db_path: Location of SQLite database for metadata.
        """

        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.conn = sqlite3.connect(db_path)
        self._setup_db()

        # Tiny MLP used as a completion stub after recall.
        hidden = max(4, dim)
        rng = np.random.default_rng(0)
        self.W1 = rng.standard_normal((dim, hidden)).astype("float32") / np.sqrt(dim)
        self.b1 = np.zeros(hidden, dtype="float32")
        self.W2 = rng.standard_normal((hidden, dim)).astype("float32") / np.sqrt(hidden)
        self.b2 = np.zeros(dim, dtype="float32")

    # ------------------------------------------------------------------
    # SQLite helpers
    def _setup_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY, value TEXT, key BLOB)"
        )
        self.conn.commit()

    def _insert_trace(self, key: np.ndarray, value: str) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO traces(value, key) VALUES (?, ?)", (value, key.tobytes()))
        self.conn.commit()
        return int(cur.lastrowid)

    def _get_trace(self, idx: int) -> Optional[tuple[str, np.ndarray]]:
        cur = self.conn.cursor()
        cur.execute("SELECT value, key FROM traces WHERE id=?", (idx,))
        row = cur.fetchone()
        if not row:
            return None
        value, key_blob = row
        key = np.frombuffer(key_blob, dtype="float32")
        return value, key

    def keys(self) -> np.ndarray:
        """Return all stored key vectors."""

        cur = self.conn.cursor()
        cur.execute("SELECT key FROM traces")
        rows = cur.fetchall()
        if not rows:
            return np.empty((0, self.dim), dtype="float32")
        return np.vstack([np.frombuffer(r[0], dtype="float32") for r in rows])

    # ------------------------------------------------------------------
    # Public API
    def write(self, key: np.ndarray, value: str) -> int:
        """Store a key/value pair.

        Args:
            key: Key vector of shape ``(dim,)``.
            value: Arbitrary metadata to associate with the key.

        Returns:
            The integer ID of the stored trace.
        """

        key = np.asarray(key, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(key)
        idx = self._insert_trace(key[0], value)
        ids = np.array([idx], dtype="int64")
        self.index.add_with_ids(key, ids)
        return idx

    def recall(self, query: np.ndarray, k: int) -> List[Trace]:
        """Recall the ``k`` nearest traces for a query vector."""

        if self.index.ntotal == 0:
            return []

        query = np.asarray(query, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query)
        scores, ids = self.index.search(query, k)
        traces: List[Trace] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            trace = self._get_trace(int(idx))
            if trace is None:
                continue
            value, key_vec = trace
            traces.append(Trace(id=int(idx), value=value, key=key_vec, score=float(score)))
        return traces

    def delete(self, idx: int) -> None:
        """Delete a trace by ``idx``."""

        ids = np.array([idx], dtype="int64")
        self.index.remove_ids(ids)
        cur = self.conn.cursor()
        cur.execute("DELETE FROM traces WHERE id=?", (idx,))
        self.conn.commit()

    def update(
        self, idx: int, key: Optional[np.ndarray] = None, value: Optional[str] = None
    ) -> None:
        """Update the key and/or value for an existing trace."""

        cur = self.conn.cursor()
        if value is not None:
            cur.execute("UPDATE traces SET value=? WHERE id=?", (value, idx))
        if key is not None:
            key_arr = np.asarray(key, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(key_arr)
            ids = np.array([idx], dtype="int64")
            self.index.remove_ids(ids)
            self.index.add_with_ids(key_arr, ids)
            cur.execute("UPDATE traces SET key=? WHERE id=?", (key_arr[0].tobytes(), idx))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Completion stub
    def complete(self, query: np.ndarray, k: int = 1) -> np.ndarray:
        """Return a simple MLP-based completion of ``query`` using recalled keys."""

        traces = self.recall(query, k)
        if not traces:
            return np.asarray(query, dtype="float32")
        mean_key = np.mean(np.stack([t.key for t in traces]), axis=0)
        h = np.tanh(mean_key @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        return out
