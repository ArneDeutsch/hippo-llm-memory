"""Utility helpers for persisted stores."""

from pathlib import Path


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
