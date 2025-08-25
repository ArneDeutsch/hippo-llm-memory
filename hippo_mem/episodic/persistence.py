"""Persistence helpers for :class:`EpisodicStore`."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .db import TraceDB
from .types import TraceValue


class TracePersistence:
    """High-level wrapper around :class:`TraceDB` handling serialization."""

    def __init__(self, path: str) -> None:
        self.db = TraceDB(path)

    def insert(self, key: np.ndarray, value: TraceValue, ts: float, salience: float) -> int:
        return self.db.insert(key, json.dumps(asdict(value)), ts, salience)

    def get(self, idx: int) -> Optional[Tuple[TraceValue, np.ndarray, float, float]]:
        row = self.db.get(idx)
        if row is None:
            return None
        value_json, key_vec, ts, salience = row
        value = TraceValue(**json.loads(value_json))
        return value, key_vec, ts, salience

    def delete(self, idx: int) -> None:
        self.db.delete(idx)

    def update_value(self, idx: int, value: TraceValue) -> None:
        self.db.update_value(idx, json.dumps(asdict(value)))

    def update_key(self, idx: int, key: np.ndarray) -> None:
        self.db.update_key(idx, key)

    def keys(self, dim: int) -> np.ndarray:
        return self.db.keys(dim)

    def all(self) -> List[Tuple[int, TraceValue, np.ndarray, float, float]]:
        """Return all persisted traces.

        Returns
        -------
        list[tuple[int, TraceValue, np.ndarray, float, float]]
            Tuples of ``(id, value, key, ts, salience)``.
        """

        cur = self.db.conn.cursor()
        cur.execute("SELECT id, value, key, ts, salience FROM traces")
        rows = []
        for idx, value_json, key_blob, ts, salience in cur.fetchall():
            value = TraceValue(**json.loads(value_json))
            key = np.frombuffer(key_blob, dtype="float32")
            rows.append((int(idx), value, key, float(ts), float(salience)))
        return rows

    # Maintenance helpers -------------------------------------------------
    def decay(self, factor: float) -> List[Tuple[int, float]]:
        return self.db.decay(factor)

    def fetch_prune_candidates(
        self, min_salience: float, cutoff: Optional[float]
    ) -> List[Sequence[object]]:
        return self.db.fetch_prune_candidates(min_salience, cutoff)

    def restore_salience(self, rows: Sequence[Tuple[int, float]]) -> None:
        self.db.restore_salience(rows)

    def restore_rows(self, rows: Sequence[Sequence[object]]) -> None:
        self.db.restore_rows(rows)


__all__ = ["TracePersistence"]
