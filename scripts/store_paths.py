"""Helpers to derive store layout from a ``RUN_ID``.

These utilities mirror the shell prelude in :mod:`scripts._env` and
provide a single source of truth for where persisted memory stores live
and what session identifier to use for replay.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
        read from ``RUN_ID`` or ``DATE`` environment variables.
    algo:
        Memory algorithm key, e.g. ``"hei_nw"``.

    Returns
    -------
    StoreLayout
        Dataclass containing base directory, algorithm subdirectory and
        deterministic session identifier. The session identifier uses the
        first segment of ``algo`` (e.g. ``hei`` for ``hei_nw``) to match
        the shell prelude's ``HEI_SESSION_ID`` variable.
    """

    rid = run_id or os.environ.get("RUN_ID") or os.environ.get("DATE")
    if not rid:
        raise ValueError("RUN_ID is required (set RUN_ID env or pass run_id argument).")

    base = Path("runs") / rid / "stores"
    algo_dir = base / algo
    prefix = algo.split("_")[0]
    session_id = f"{prefix}_{rid}"
    return StoreLayout(base, algo_dir, session_id)


__all__ = ["StoreLayout", "derive"]
