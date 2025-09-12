# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from pathlib import Path

import pytest

from hippo_mem.utils.stores import validate_store


def test_validate_store_explicit(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    algo_dir = base / "hei_nw"
    session_dir = algo_dir / "hei_run"
    session_dir.mkdir(parents=True)
    (session_dir / "episodic.jsonl").write_text("{}\n")
    path = validate_store(
        run_id="run",
        preset="memory/hei_nw",
        algo="hei_nw",
        kind="episodic",
        store_dir=str(base),
        session_id="hei_run",
    )
    assert path == session_dir / "episodic.jsonl"


def test_validate_store_missing(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    with pytest.raises(FileNotFoundError) as err:
        validate_store(
            run_id="run",
            preset="memory/hei_nw",
            algo="hei_nw",
            kind="episodic",
            store_dir=str(base),
            session_id="hei_run",
        )
    assert "Persisted store not found" in str(err.value)


def test_validate_store_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rid = "r1"
    path = validate_store(run_id=rid, preset="baselines/core", algo="hei_nw", kind="episodic")
    assert path is None
