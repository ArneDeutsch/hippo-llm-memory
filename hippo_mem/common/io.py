# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Thread-safe I/O helpers with atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

# per-path locks to guard concurrent access within a process
_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_lock(path: Path) -> threading.Lock:
    """Return a lock for ``path`` shared across threads."""

    key = str(path)
    with _locks_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def atomic_write_file(path: str | Path, writer: Callable[[Path], None]) -> None:
    """Atomically write to ``path`` using ``writer``.

    The ``writer`` callback receives a temporary path. It should write the
    desired content to that location. The temp file is then ``os.replace``d
    to the target path, ensuring atomicity on POSIX systems.
    """

    target = Path(path)
    lock = _get_lock(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=target.parent)
        tmp_path = Path(tmp.name)
        try:
            tmp.close()
            writer(tmp_path)
            os.replace(tmp_path, target)
        finally:
            if tmp_path.exists():  # pragma: no cover - cleanup safety
                tmp_path.unlink(missing_ok=True)


def atomic_write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    """Atomically write ``records`` as JSONL to ``path``."""

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    atomic_write_file(path, _write)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Atomically write ``obj`` as JSON to ``path``."""

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    atomic_write_file(path, _write)


def read_jsonl(path: str | Path) -> Iterator[Any]:
    """Yield JSON objects from ``path`` under a thread lock."""

    file = Path(path)
    lock = _get_lock(file)
    with lock:
        with open(file, "r", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def read_json(path: str | Path) -> Any:
    """Read a JSON file under a thread lock."""

    file = Path(path)
    lock = _get_lock(file)
    with lock:
        with open(file, "r", encoding="utf-8") as fh:
            return json.load(fh)


def read_parquet(path: str | Path):  # type: ignore[override]
    """Read a Parquet file under a thread lock."""

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional
        raise RuntimeError("Parquet support requires pandas") from exc
    file = Path(path)
    lock = _get_lock(file)
    with lock:
        return pd.read_parquet(file)


__all__ = [
    "atomic_write_file",
    "atomic_write_json",
    "atomic_write_jsonl",
    "read_json",
    "read_jsonl",
    "read_parquet",
]
