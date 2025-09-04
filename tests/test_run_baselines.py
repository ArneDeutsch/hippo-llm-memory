import json
from pathlib import Path

import pytest

from hippo_eval.baselines import aggregate_metrics
from hippo_eval.harness.io import write_metrics


def _write_metrics(
    tmp_path: Path, preset: str, suite: str, seed: int, em_raw: float, em_norm: float, f1: float
) -> None:
    run_dir = tmp_path / "runs" / "20250101" / "baselines" / preset / suite / f"50_{seed}"
    run_dir.mkdir(parents=True)
    metrics = {"metrics": {suite: {"pre_em_raw": em_raw, "pre_em_norm": em_norm, "pre_f1": f1}}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_collect_and_write(tmp_path: Path) -> None:
    _write_metrics(tmp_path, "core", "episodic", 0, 0.1, 0.2, 0.3)
    _write_metrics(tmp_path, "core", "episodic", 1, 0.2, 0.3, 0.4)
    root = tmp_path / "runs" / "20250101" / "baselines"
    rows = aggregate_metrics(root)
    assert rows == [
        {
            "suite": "episodic",
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
    csv_path = write_metrics(rows, out_dir)
    assert csv_path.exists()
    assert (out_dir / "baselines_ok.flag").exists()
