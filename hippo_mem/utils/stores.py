# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Utility helpers for persisted stores and preset checks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class StoreLayout:
    """Resolved store paths for a given ``RUN_ID`` and algorithm."""

    base_dir: Path
    algo_dir: Path
    session_id: str


def derive(run_id: str | None = None, algo: str = "hei_nw") -> StoreLayout:
    """Derive store paths and session identifier.

    Parameters
    ----------
    run_id:
        Identifier for the current experiment run. If ``None`` the value is
        read from the ``RUN_ID`` environment variable.
    algo:
        Memory algorithm key, e.g. ``"hei_nw"``.

    Returns
    -------
    StoreLayout
        Dataclass containing base directory, algorithm subdirectory and
        deterministic session identifier. The session identifier uses the
        first segment of ``algo`` (e.g. ``hei`` for ``hei_nw``) to match the
        shell prelude's ``HEI_SESSION_ID`` variable.
    """

    rid = run_id or os.environ.get("RUN_ID")
    if not rid:
        raise ValueError("RUN_ID is required (set RUN_ID env or pass run_id argument).")

    base = Path("runs") / rid / "stores"
    algo_dir = base / algo
    prefix = algo.split("_")[0]
    session_id = f"{prefix}_{rid}"
    return StoreLayout(base, algo_dir, session_id)


def is_memory_preset(preset: str | None) -> bool:
    """Return ``True`` if ``preset`` denotes a memory preset."""

    if not preset:
        return False
    return "memory" in str(preset).split("/")


def assert_store_exists(store_dir: str, session_id: str, algo: str, kind: str = "episodic") -> Path:
    """Assert that a persisted store exists for a given algorithm.

    Parameters
    ----------
    store_dir : str
        Base directory containing all stores **without** the trailing algorithm
        subfolder. Convenience wrappers handle appending this suffix before
        calling.
    session_id : str
        Session identifier.
    algo : str
        Algorithm identifier (e.g. ``"hei_nw"`` or ``"sgc_rss"``).
    kind : str, optional
        Store kind, by default ``"episodic"``.

    Returns
    -------
    pathlib.Path
        Path to the store file.
    """

    p = Path(store_dir) / algo / session_id / f"{kind}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            "Persisted store not found.\n"
            f"Expected path: {p}\n"
            f"Hint: `store_dir` should be the base directory containing the `{algo}` folder.\n"
            "Reminder: run teach+replay with `persist=true` to create it."
        )
    return p


def scan_episodic_store(path: Path) -> Tuple[int, int]:
    """Return trace count and number of non-zero keys."""

    count = nz = 0
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("key") or []
                if any(abs(float(v)) > 1e-12 for v in key):
                    nz += 1
                count += 1
    else:
        from hippo_mem.episodic.persistence import TracePersistence

        tp = TracePersistence(str(path))
        for _idx, _val, key, _ts, _sal in tp.all():
            count += 1
            if float(np.linalg.norm(key)) > 1e-12:
                nz += 1
    return count, nz


def scan_kg_store(path: Path) -> Tuple[int, int, int, int]:
    """Return node and edge counts plus non-zero embedding totals."""

    nodes = edges = node_nz = edge_nz = 0
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                typ = rec.get("type")
                emb = rec.get("embedding")
                has_emb = emb and any(abs(float(v)) > 1e-12 for v in emb)
                if typ == "node":
                    nodes += 1
                    if has_emb:
                        node_nz += 1
                elif typ == "edge":
                    edges += 1
                    if has_emb:
                        edge_nz += 1
    else:
        from hippo_mem.relational.backend import SQLiteBackend

        backend = SQLiteBackend(str(path))
        node_rows = backend.exec("SELECT embedding FROM nodes", fetch="all") or []
        nodes = len(node_rows)
        for (emb_json,) in node_rows:
            if emb_json:
                emb = json.loads(emb_json)
                if any(abs(float(v)) > 1e-12 for v in emb):
                    node_nz += 1
        edge_rows = backend.exec("SELECT embedding FROM edges", fetch="all") or []
        edges = len(edge_rows)
        for (emb_json,) in edge_rows:
            if emb_json:
                emb = json.loads(emb_json)
                if any(abs(float(v)) > 1e-12 for v in emb):
                    edge_nz += 1
    return nodes, edges, node_nz, edge_nz


def validate_store(
    run_id: str,
    preset: str,
    algo: str,
    kind: str = "episodic",
    store_dir: str | None = None,
    session_id: str | None = None,
) -> Path | None:
    """Resolve the expected store path and assert it exists.

    Prefer explicit ``store_dir``/``session_id`` when provided; otherwise derive
    the layout from ``run_id`` and ``algo``.
    """

    if store_dir and session_id:
        base = Path(store_dir)
        algo_dir = base if base.name == algo else base / algo
        filename = {
            "episodic": "episodic.jsonl",
            "kg": "kg.jsonl",
            "spatial": "spatial.jsonl",
        }[kind]
        path = algo_dir / session_id / filename
        if path.exists():
            return path
        alt = base / session_id / filename
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Persisted store not found. Expected path: {path}")

    layout = derive(run_id=run_id, algo=algo)
    if preset and not is_memory_preset(preset):
        if layout.algo_dir.exists():
            raise FileExistsError(
                f"unexpected store directory for baseline preset {preset}: {layout.algo_dir}"
            )
        return None
    return assert_store_exists(str(layout.base_dir), layout.session_id, algo, kind=kind)


__all__ = [
    "StoreLayout",
    "derive",
    "is_memory_preset",
    "assert_store_exists",
    "scan_episodic_store",
    "scan_kg_store",
    "validate_store",
]
