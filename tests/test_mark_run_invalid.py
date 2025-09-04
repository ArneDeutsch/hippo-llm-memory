import json
import subprocess
import sys
from pathlib import Path

from scripts.tools.mark_run_invalid import mark_run_invalid


def test_mark_run_invalid_skips_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "20250101"
    metrics_dir = run_dir / "baselines" / "core" / "episodic" / "50_1337"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(json.dumps({"replay": {"samples": 0}}))

    mark_run_invalid(run_dir)
    assert (run_dir / "INVALID").exists()

    reports_dir = tmp_path / "reports"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hippo_eval.reporting.report",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--out-dir",
            str(reports_dir),
            "--run-id",
            "20250101",
        ],
        check=True,
    )
    assert not (reports_dir / "20250101" / "index.md").exists()
