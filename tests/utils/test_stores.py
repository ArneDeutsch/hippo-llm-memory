# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from pathlib import Path

import pytest

from hippo_mem.utils import stores


def test_derive_requires_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_ID", raising=False)
    with pytest.raises(ValueError):
        stores.derive()


def test_assert_store_exists_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        stores.assert_store_exists(str(tmp_path), "sid", "hei_nw")


def test_assert_store_exists_ok(tmp_path: Path) -> None:
    p = tmp_path / "hei_nw" / "sid" / "episodic.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("{}\n")
    assert stores.assert_store_exists(str(tmp_path), "sid", "hei_nw") == p


def test_scan_episodic_store_counts(tmp_path: Path) -> None:
    file = tmp_path / "episodic.jsonl"
    file.write_text('{"key": [1.0]}\n{}\n')
    count, nz = stores.scan_episodic_store(file)
    assert count == 2
    assert nz == 1


def test_is_memory_preset() -> None:
    assert stores.is_memory_preset("memory/hei_nw")
    assert not stores.is_memory_preset("baseline")
