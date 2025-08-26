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
    """Sample replay items from saved stores.

    Parameters
    ----------
    store_dir : str
        Base directory containing persisted stores.
    session_id : str
        Identifier of the session to load.
    ratios : dict, optional
        Sampling ratios for ``{"episodic", "relational", "spatial"}``, by
        default ``{"episodic": 0.6, "relational": 0.3, "spatial": 0.1}``.
    seed : int, optional
        Random seed, by default ``0``.
    teacher : callable, optional
        Function called as ``teacher(record)`` returning a mapping with
        optional ``{"text", "logits"}`` fields. When provided and a sampled
        record lacks a ``"teacher"`` entry, the function is invoked with the
        record and the returned value is stored under ``record["teacher"]``.
        This enables optional distillation from a teacher model run with
        memory enabled.
    policy : {"uniform", "priority", "spaced"}, optional
        Sampling policy. ``"priority"`` uses precomputed weights, ``"uniform"``
        ignores them and ``"spaced"`` iterates without replacement per
        ``cycle``. Default is ``"priority"``.
    cycles : int, optional
        Number of full passes over the dataset. ``None`` yields an infinite
        iterator. Default is ``None``.
    noise_level : float, optional
        Standard deviation of Gaussian noise added to weights for
        ``policy="priority"``. Default ``0.0`` means deterministic weights.
    max_items : int, optional
        Maximum number of items to yield. Takes precedence over ``cycles``
        when both are set.
    """

    def __init__(
        self,
        store_dir: str,
        session_id: str,
        *,
        ratios: Optional[Dict[str, float]] = None,
        seed: int = 0,
        teacher: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        policy: str = "priority",
        cycles: Optional[int] = None,
        noise_level: float = 0.0,
        max_items: Optional[int] = None,
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
        self.policy = policy
        self.cycles = cycles
        self.noise_level = noise_level
        self.max_items = max_items
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

        total_per_cycle = sum(len(data.records) for data in stores.values())
        max_items = self.max_items
        if self.cycles is not None:
            cyc_total = self.cycles * total_per_cycle
            max_items = min(cyc_total, max_items) if max_items is not None else cyc_total

        count = 0

        if self.policy == "spaced":
            # prepare shuffled indices per store
            orders = {k: rng.permutation(len(data.records)).tolist() for k, data in stores.items()}
            pos = {k: 0 for k in kinds}
            cycle = 0
            while max_items is None or count < max_items:
                available = [k for k in kinds if pos[k] < len(orders[k])]
                if not available:
                    cycle += 1
                    if self.cycles is not None and cycle >= self.cycles:
                        break
                    for k in kinds:
                        orders[k] = rng.permutation(len(stores[k].records)).tolist()
                        pos[k] = 0
                    available = kinds
                p = np.array([self.ratios.get(k, 0.0) for k in available], dtype="float64")
                p = p / p.sum()
                kind = rng.choice(available, p=p)
                idx = orders[kind][pos[kind]]
                pos[kind] += 1

                data = stores[kind]
                base_rec = data.records[idx]

                if self.teacher is not None and "teacher" not in base_rec:
                    teacher_out = self.teacher(base_rec)
                    if teacher_out is not None:
                        base_rec["teacher"] = teacher_out

                rec = dict(base_rec)
                rec["kind"] = kind
                yield rec
                count += 1
        else:
            while max_items is None or count < max_items:
                kind = rng.choice(kinds, p=probs)
                data = stores[kind]
                if self.policy == "priority" and len(data.records) > 0:
                    weights = data.weights
                    if self.noise_level > 0.0 and len(weights) > 0:
                        noise = rng.normal(0.0, self.noise_level, size=len(weights))
                        weights = np.clip(weights + noise, 0.0, None)
                        if weights.sum() == 0:
                            weights = np.ones_like(weights) / len(weights)
                        else:
                            weights = weights / weights.sum()
                    idx = int(rng.choice(len(data.records), p=weights))
                else:  # uniform
                    idx = int(rng.integers(len(data.records)))

                base_rec = data.records[idx]
                if self.teacher is not None and "teacher" not in base_rec:
                    teacher_out = self.teacher(base_rec)
                    if teacher_out is not None:
                        base_rec["teacher"] = teacher_out

                rec = dict(base_rec)
                rec["kind"] = kind
                yield rec
                count += 1


__all__ = ["ReplayDataset"]
