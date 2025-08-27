import csv
import json
import subprocess
import sys
from pathlib import Path


def test_metrics_csv_has_diagnostics(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=baselines/core",
        "n=2",
        "seed=1337",
        f"outdir={outdir}",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)

    csv_path = outdir / "metrics.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    for col in ["pred_len", "gold_len", "overlong", "format_violation"]:
        assert col in headers

    metrics = json.loads((outdir / "metrics.json").read_text())
    diag = metrics["diagnostics"]["episodic"]
    assert "pre_overlong" in diag
    assert "pre_format_violation" in diag
