# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import os
from pathlib import Path

from scripts.eval_model import _normalize_store_dir


def test_normalize_base(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    assert _normalize_store_dir(str(base), "hei_nw") == str(base)


def test_normalize_with_algo(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    path = base / "hei_nw"
    assert _normalize_store_dir(str(path), "hei_nw") == str(base)


def test_normalize_mixed_separators(tmp_path: Path) -> None:
    base = tmp_path / "stores"
    mixed = os.path.join(str(base), "hei_nw")
    assert _normalize_store_dir(mixed, "hei_nw") == str(base)
