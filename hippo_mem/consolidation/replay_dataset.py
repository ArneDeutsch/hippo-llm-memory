"""Replay dataset built from persisted stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import torch

import hippo_mem.common.io as io


@dataclass
class _StoreData:
    records: List[Dict[str, Any]]
    weights: np.ndarray


class ReplayDataset(torch.utils.data.IterableDataset):
    """Sample replay items from saved stores with priority weighting.

    Parameters
    ----------
    store_dir : str
        Base directory containing persisted stores.
    session_id : str
        Identifier of the session to load.
    ratios : dict, optional
        Sampling ratios for ``{"episodic", "relational", "spatial"}``,
        by default ``{"episodic": 0.6, "relational": 0.3, "spatial": 0.1}``.
    seed : int, optional
        Random seed, by default ``0``.
    teacher : callable, optional
        Function called as ``teacher(record)`` returning a mapping with
        optional ``{"text", "logits"}`` fields. When provided and a
        sampled record lacks a ``"teacher"`` entry, the function is invoked
        with the record and the returned value is stored under
        ``record["teacher"]``. This enables optional distillation from a
        teacher model run with memory enabled.
    """

    def __init__(
        self,
        store_dir: str,
        session_id: str,
        *,
        ratios: Optional[Dict[str, float]] = None,
        seed: int = 0,
        teacher: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__()
        self.store_dir = store_dir
        self.session_id = session_id
        self.ratios = ratios or {
            "episodic": 0.6,
            "relational": 0.3,
            "spatial": 0.1,
        }
        self.seed = seed
        self.teacher = teacher
        self._stores: Dict[str, _StoreData] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        base = Path(self.store_dir) / self.session_id
        self._stores["episodic"] = self._load_episodic(base / "episodic.jsonl")
        self._stores["relational"] = self._load_generic(
            base / "relational.jsonl",
            lambda rec: rec.get("type") == "edge",
        )
        self._stores["spatial"] = self._load_generic(
            base / "spatial.jsonl", lambda rec: rec.get("type") == "edge"
        )

    def _load_episodic(self, path: Path) -> _StoreData:
        records: List[Dict[str, Any]] = []
        ts_max = 0.0
        if path.exists():
            for rec in io.read_jsonl(path):
                records.append(rec)
                ts_max = max(ts_max, float(rec.get("ts", 0.0)))
        weights: List[float] = []
        for rec in records:
            sal = float(rec.get("salience", 1.0))
            usage = float(rec.get("usage", 0.0))
            ts = float(rec.get("ts", ts_max))
            novelty = 1.0 / (1.0 + (ts_max - ts))
            w = sal * novelty / (1.0 + usage)
            weights.append(w)
        arr = np.asarray(weights, dtype="float64")
        if arr.sum() > 0:
            arr /= arr.sum()
        return _StoreData(records, arr)

    def _load_generic(
        self, path: Path, predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> _StoreData:
        records: List[Dict[str, Any]] = []
        if path.exists():
            for rec in io.read_jsonl(path):
                if predicate is None or predicate(rec):
                    records.append(rec)
        if records:
            arr = np.ones(len(records), dtype="float64")
            arr /= arr.sum()
        else:
            arr = np.asarray([], dtype="float64")
        return _StoreData(records, arr)

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        kinds = [k for k, data in self._stores.items() if data.records]
        if not kinds:
            return iter(())  # type: ignore[return-value]
        probs = np.array([self.ratios.get(k, 0.0) for k in kinds], dtype="float64")
        probs = probs / probs.sum()
        stores = {k: self._stores[k] for k in kinds}
        while True:
            kind = rng.choice(kinds, p=probs)
            data = stores[kind]
            idx = int(rng.choice(len(data.records), p=data.weights))
            base_rec = data.records[idx]

            if self.teacher is not None and "teacher" not in base_rec:
                teacher_out = self.teacher(base_rec)
                if teacher_out is not None:
                    base_rec["teacher"] = teacher_out

            rec = dict(base_rec)
            rec["kind"] = kind
            yield rec


__all__ = ["ReplayDataset"]
