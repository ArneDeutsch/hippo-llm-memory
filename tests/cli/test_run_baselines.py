import json
from pathlib import Path

import pytest

from hippo_eval.baselines import aggregate_metrics
from hippo_eval.eval.writers import write_baseline_metrics


def _write_metrics(
    tmp_path: Path, preset: str, suite: str, seed: int, em_raw: float, em_norm: float, f1: float
) -> None:
    run_dir = tmp_path / "runs" / "20250101" / "baselines" / preset / suite / f"50_{seed}"
    run_dir.mkdir(parents=True)
    metrics = {"metrics": {suite: {"pre_em_raw": em_raw, "pre_em_norm": em_norm, "pre_f1": f1}}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_collect_and_write(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "core", "episodic_cross_mem", 0, 0.1, 0.2, 0.3)
    _write_metrics(tmp_path, "core", "episodic_cross_mem", 1, 0.2, 0.3, 0.4)
    root = tmp_path / "runs" / "20250101" / "baselines"
    rows = aggregate_metrics(root)
    assert rows == [
        {
            "suite": "episodic_cross_mem",
            "preset": "core",
            "em_raw_mean": 0.15000000000000002,
            "em_raw_ci": pytest.approx(0.069, abs=1e-3),
            "em_norm_mean": 0.25,
            "em_norm_ci": pytest.approx(0.069, abs=1e-3),
            "f1_mean": 0.35,
            "f1_ci": pytest.approx(0.069, abs=1e-3),
        }
    ]
    out_dir = root
    csv_path = write_baseline_metrics(rows, out_dir)
    assert csv_path.exists()
    assert (out_dir / "baselines_ok.flag").exists()


def test_collect_metrics_flat_layout(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic_cross_mem"
    root.mkdir(parents=True)
    metrics = {
        "metrics": {"episodic_cross_mem": {"pre_em_raw": 0.3, "pre_em_norm": 0.4, "pre_f1": 0.5}}
    }
    (root / "metrics.json").write_text(json.dumps(metrics))
    rows = aggregate_metrics(root.parent.parent)
    assert rows == [
        {
            "suite": "episodic_cross_mem",
            "preset": "core",
            "em_raw_mean": 0.3,
            "em_raw_ci": 0.0,
            "em_norm_mean": 0.4,
            "em_norm_ci": 0.0,
            "f1_mean": 0.5,
            "f1_ci": 0.0,
        }
    ]
