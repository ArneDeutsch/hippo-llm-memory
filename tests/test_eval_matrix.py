import json
import subprocess
import sys
from pathlib import Path


def test_run_matrix_generates_outputs(tmp_path: Path) -> None:
    """Matrix run writes metrics for each combination."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_bench.py"
    outdir = tmp_path / "matrix_runs"
    cmd = [
        sys.executable,
        str(script),
        "+run_matrix=true",
        "preset=baselines/core",
        "+suites=[episodic]",
        "+n_values=[2]",
        "+seeds=[123]",
        f"outdir={outdir}",
    ]
    subprocess.run(cmd, check=True)
    expected = outdir / "episodic" / "n2_seed123"
    assert (expected / "metrics.json").exists()
    with (expected / "metrics.json").open() as f:
        metrics = json.load(f)
    assert metrics["n"] == 2
    assert metrics["seed"] == 123
