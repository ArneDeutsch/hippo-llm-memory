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

    # ------------------------------------------------------------------
    # SQLite helpers
    def _setup_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY, value TEXT)")
        self.conn.commit()

    def _insert_value(self, value: str) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO traces(value) VALUES (?)", (value,))
        self.conn.commit()
        return int(cur.lastrowid)

    def _get_value(self, idx: int) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM traces WHERE id=?", (idx,))
        row = cur.fetchone()
        return row[0] if row else None

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
        idx = self._insert_value(value)
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
            value = self._get_value(int(idx))
            if value is None:
                continue
            traces.append(Trace(id=int(idx), value=value, score=float(score)))
        return traces
