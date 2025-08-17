"""Episodic memory store backed by FAISS and SQLite."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Union

import faiss  # type: ignore
import numpy as np

from .db import TraceDB
from .gating import DGKey, densify, k_wta
from .index import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class TraceValue:
    """Metadata associated with a stored trace."""

    tokens_span: Optional[tuple[int, int]] = None
    entity_slots: Optional[dict] = None
    state_sketch: Optional[list] = None
    salience_tags: Optional[List[str]] = None
    provenance: Optional[str] = None


@dataclass
class Trace:
    """A retrieved memory trace."""

    id: int
    value: TraceValue
    key: DGKey
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
        k_wta: int = 0,
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
        self.k_wta = k_wta
        self.index = VectorIndex(dim, index_str, train_threshold)
        self.db = TraceDB(db_path)

        # Hopfield parameter (inverse temperature)
        self.beta = 1.0

        # Configuration and logging
        self.config = config or {}
        self._log = {"writes": 0, "recalls": 0, "hits": 0, "maintenance": 0}
        self._bg_thread: Optional[threading.Thread] = None
        self._history: List[dict[str, Any]] = []
        self._max_undo = int(self.config.get("max_undo", 5))
        self._maintenance_log: List[dict[str, Any]] = []
        self._log_file = self.config.get("maintenance_log")

    # ------------------------------------------------------------------
    def keys(self) -> np.ndarray:
        """Return all stored key vectors."""

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
        """k-Winner-Take-All sparse encoding of ``query``."""

        return k_wta(query, k)

    def _to_dense(self, key: DGKey) -> np.ndarray:
        """Convert ``key`` back to a dense vector."""

        return densify(key)

    # ------------------------------------------------------------------
    # Public API
    def write(self, key: Union[np.ndarray, DGKey], value: TraceValue | str) -> int:
        """Store a key/value pair."""

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
        salience = float(len(value.salience_tags) if value.salience_tags else 1.0)
        idx = self.db.insert(key_arr[0], json.dumps(asdict(value)), ts, salience)
        self.index.add(key_arr, idx)
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
        return traces

    def delete(self, idx: int) -> None:
        """Delete a trace by ``idx``."""

        self.index.remove(idx)
        self.db.delete(idx)

    def update(
        self,
        idx: int,
        key: Optional[Union[np.ndarray, DGKey]] = None,
        value: Optional[TraceValue] = None,
    ) -> None:
        """Update the key and/or value for an existing trace."""

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
        """Return a Hopfield-based completion of ``query`` using recalled keys."""
        if not self.config.get("hopfield", True):
            return np.asarray(query, dtype="float32")

        traces = self.recall(query, k)
        if not traces:
            return np.asarray(query, dtype="float32")
        patterns = np.stack([self._to_dense(t.key) for t in traces])
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
        rows = self.db.decay(factor)
        if rows:
            self._push_history("decay", rows)
        self._log_event("decay", {"rate": rate})

    def prune(self, min_salience: float = 0.1, max_age: Optional[float] = None) -> None:
        """Remove traces whose salience or age fall below thresholds."""

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
        """Rollback the last ``n`` maintenance operations."""

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
                    key_vec = np.frombuffer(key_blob, dtype="float32").reshape(1, -1)
                    faiss.normalize_L2(key_vec)
                    self.index.add(key_vec, int(idx))
            self._log_event("rollback", {"op": op})


__all__ = ["TraceValue", "Trace", "EpisodicStore"]
