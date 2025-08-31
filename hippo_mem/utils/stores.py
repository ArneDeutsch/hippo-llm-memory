"""Utility helpers for persisted stores and preset checks."""

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
        first segment of ``algo`` (e.g. ``hei`` for ``hei_nw``) to match the
        shell prelude's ``HEI_SESSION_ID`` variable.
    """

    rid = run_id or os.environ.get("RUN_ID") or os.environ.get("DATE")
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


def validate_store(
    run_id: str | None = None,
    algo: str = "hei_nw",
    kind: str = "episodic",
    preset: str | None = None,
) -> Path | None:
    """Validate expected persisted store layout before replay.

    For memory presets this asserts that the store file exists and returns its
    path. For baseline presets it verifies that no store directory has been
    created and returns ``None``.
    """

    layout = derive(run_id=run_id, algo=algo)
    if preset and not is_memory_preset(preset):
        if layout.algo_dir.exists():
            raise FileExistsError(
                f"unexpected store directory for baseline preset {preset}: {layout.algo_dir}"
            )
        return None
    return assert_store_exists(str(layout.base_dir), layout.session_id, algo, kind=kind)


__all__ = ["StoreLayout", "derive", "is_memory_preset", "assert_store_exists", "validate_store"]
