"""Store path helpers used by evaluation harness and validators."""

from __future__ import annotations

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


__all__ = ["resolve_store_meta_path"]
