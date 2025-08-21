"""SQLite helpers for the episodic store."""

from __future__ import annotations

import sqlite3
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hippo_mem.common.sqlite import SQLiteExecMixin


class TraceDB(SQLiteExecMixin):
    """Lightweight wrapper around the SQLite trace table.

    The class exposes a tiny subset of operations used by :class:`EpisodicStore`
    so that SQLite interactions can be unit tested independently from FAISS
    operations."""

    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._setup()

    # ------------------------------------------------------------------
    def _setup(self) -> None:
        self._exec(
            "CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY, value TEXT, key BLOB, ts REAL, salience REAL)"
        )

    # ------------------------------------------------------------------
    def insert(self, key: np.ndarray, value_json: str, ts: float, salience: float) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO traces(value, key, ts, salience) VALUES (?, ?, ?, ?)",
            (value_json, key.tobytes(), ts, salience),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get(self, idx: int) -> Optional[Tuple[str, np.ndarray, float, float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT value, key, ts, salience FROM traces WHERE id=?", (idx,))
        row = cur.fetchone()
        if not row:
            return None
        value, key_blob, ts, salience = row
        key = np.frombuffer(key_blob, dtype="float32")
        return value, key, float(ts), float(salience)

    def delete(self, idx: int) -> None:
        self._exec("DELETE FROM traces WHERE id=?", (idx,))

    def update_value(self, idx: int, value_json: str) -> None:
        self._exec("UPDATE traces SET value=? WHERE id=?", (value_json, idx))

    def update_key(self, idx: int, key: np.ndarray) -> None:
        self._exec("UPDATE traces SET key=? WHERE id=?", (key.tobytes(), idx))

    def keys(self, dim: int) -> np.ndarray:
        cur = self.conn.cursor()
        cur.execute("SELECT key FROM traces")
        rows = cur.fetchall()
        if not rows:
            return np.empty((0, dim), dtype="float32")
        return np.vstack([np.frombuffer(r[0], dtype="float32") for r in rows])

    # ------------------------------------------------------------------
    def decay(self, factor: float) -> List[Tuple[int, float]]:
        """Apply exponential decay to salience values and return previous values."""

        cur = self.conn.cursor()
        cur.execute("SELECT id, salience FROM traces")
        rows = cur.fetchall()
        cur.execute("UPDATE traces SET salience = salience * ?", (factor,))
        self.conn.commit()
        return [(int(i), float(s)) for i, s in rows]

    def fetch_prune_candidates(
        self, min_salience: float, cutoff: Optional[float]
    ) -> List[Sequence[object]]:
        cur = self.conn.cursor()
        conditions: List[str] = []
        params: List[float] = []
        if min_salience is not None:
            conditions.append("salience < ?")
            params.append(min_salience)
        if cutoff is not None:
            conditions.append("ts < ?")
            params.append(cutoff)
        if not conditions:
            return []
        query = "SELECT id, value, key, ts, salience FROM traces WHERE " + " OR ".join(conditions)
        cur.execute(query, params)
        rows = cur.fetchall()
        return rows

    def restore_salience(self, rows: Sequence[Tuple[int, float]]) -> None:
        cur = self.conn.cursor()
        cur.executemany(
            "UPDATE traces SET salience=? WHERE id=?",
            [(s, i) for i, s in rows],
        )
        self.conn.commit()

    def restore_rows(self, rows: Sequence[Sequence[object]]) -> None:
        cur = self.conn.cursor()
        for idx, value_json, key_blob, ts, salience in rows:
            cur.execute(
                "INSERT OR REPLACE INTO traces(id, value, key, ts, salience) VALUES (?,?,?,?,?)",
                (idx, value_json, key_blob, ts, salience),
            )
        self.conn.commit()


__all__ = ["TraceDB"]
