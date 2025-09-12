# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Tests for the baseline aggregator CLI."""

import json
from pathlib import Path

import pytest

from hippo_eval.baselines import main as run_main


def _write_metrics(base: Path, run_id: str) -> None:
    """Write a minimal metrics.json for ``run_id`` under ``base``."""

    run_dir = base / "runs" / run_id / "baselines" / "core" / "episodic" / "50_1337"
    run_dir.mkdir(parents=True)
    metrics = {"metrics": {"episodic": {"pre_em_raw": 0.1, "pre_em_norm": 0.2, "pre_f1": 0.3}}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_aggregator_uses_run_id(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "TEST123")
    run_main(["--runs-dir", str(tmp_path / "runs"), "--run-id", "TEST123"])
    out_dir = tmp_path / "runs" / "TEST123" / "baselines"
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "baselines_ok.flag").exists()


def test_aggregator_error_lists_candidates(tmp_path: Path) -> None:
    # create a candidate run so the error lists it
    (tmp_path / "runs" / "ID1" / "baselines").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        run_main(["--runs-dir", str(tmp_path / "runs"), "--run-id", "ID2"])
    assert "ID1" in str(excinfo.value)
