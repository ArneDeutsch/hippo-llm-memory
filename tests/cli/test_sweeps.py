"""Tests for sweep pruning utility."""

import json
from pathlib import Path

from scripts import sweeps


def _make_sweep_dir(tmp_path: Path, em_raw: float, f1: float) -> Path:
    """Create mock sweep directory with metrics and meta."""
    metrics = {"metrics": {"episodic_cross_mem": {"pre_em_raw": em_raw, "pre_f1": f1}}}
    meta = {"suite": "episodic_cross_mem", "config": {"episodic": {"gate": {"tau": 0.5}}}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_process_sweep_dir_marks_non_informative(tmp_path: Path) -> None:
    """Sweep within epsilon creates marker file."""
    baselines = {"episodic_cross_mem": (0.5, 0.4)}
    path = _make_sweep_dir(tmp_path, 0.51, 0.41)
    row = sweeps.process_sweep_dir(path, baselines, eps=0.05)
    assert not row["informative"]
    assert (path / "non_informative.flag").exists()


def test_process_sweep_dir_informative(tmp_path: Path) -> None:
    """Sweep above epsilon is retained."""
    baselines = {"episodic_cross_mem": (0.5, 0.4)}
    path = _make_sweep_dir(tmp_path, 0.7, 0.4)
    row = sweeps.process_sweep_dir(path, baselines, eps=0.05)
    assert row["informative"]
    assert not (path / "non_informative.flag").exists()
