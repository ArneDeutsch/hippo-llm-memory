import json
import subprocess
import sys
from pathlib import Path


def test_eval_model_run_matrix(tmp_path: Path) -> None:
    """Matrix run writes metrics for each combination."""

    outdir = tmp_path / "matrix"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "+run_matrix=true",
        "preset=memory/hei_nw",
        "+suites=[episodic]",
        "+n_values=[2]",
        "+seeds=[1337]",
        f"outdir={outdir}",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    expected = outdir / "episodic" / "2_1337" / "metrics.json"
    assert expected.exists()
    metrics = json.loads(expected.read_text())
    assert metrics["n"] == 2
    assert metrics["seed"] == 1337
