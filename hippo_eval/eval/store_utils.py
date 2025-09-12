# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Store path helpers used by evaluation harness and validators."""

from __future__ import annotations

import copy
from pathlib import Path


def resolve_store_meta_path(preset: str, store_dir: Path, session_id: str) -> Path:
    """Return the ``store_meta.json`` path for a given preset and store directory.

    Parameters
    ----------
    preset:
        Preset identifier such as ``"memory/hei_nw"``.
    store_dir:
        Base directory passed via ``--store_dir``. This may either point to the
        run's ``stores`` directory or already include the algorithm suffix.
    session_id:
        Session identifier used by the store.
    """

    algo = preset.split("/")[-1]
    sd = Path(store_dir)
    if sd.name == algo:
        return sd / session_id / "store_meta.json"
    return sd / algo / session_id / "store_meta.json"


def fork_store(store):
    """Return a deep copy of ``store`` for isolation."""

    return copy.deepcopy(store)


def clear_store(store) -> None:
    """Remove all items from ``store`` in-place."""

    if hasattr(store, "persistence") and hasattr(store, "index"):
        try:
            store.persistence.db.conn.execute("DELETE FROM traces")
            store.persistence.db.conn.commit()
            try:
                store.index.reset()
            except Exception:
                store.index = type(store.index)(
                    store.dim,
                    getattr(store.index, "index_str", "Flat"),
                    getattr(store.index, "train_threshold", 100),
                )
        except Exception:
            pass
    if hasattr(store, "graph"):
        try:
            store.graph.clear()
        except Exception:
            store.graph = {}
    if hasattr(store, "_context_to_id"):
        store._context_to_id.clear()


__all__ = ["resolve_store_meta_path", "fork_store", "clear_store"]
