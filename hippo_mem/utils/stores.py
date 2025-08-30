from pathlib import Path


def assert_store_exists(store_dir: str, session_id: str, kind: str = "episodic") -> Path:
    """Assert that a persisted store exists.

    Parameters
    ----------
    store_dir : str
        Base directory containing all stores **without** the trailing ``hei_nw``.
        Convenience wrappers handle appending this suffix before calling.
    session_id : str
        Session identifier.
    kind : str, optional
        Store kind, by default ``"episodic"``.

    Returns
    -------
    pathlib.Path
        Path to the store file.
    """
    p = Path(store_dir) / "hei_nw" / session_id / f"{kind}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing persisted store: {p}\n"
            "Run §4.1 (teach + replay with persist=true) to create it."
        )
    return p
