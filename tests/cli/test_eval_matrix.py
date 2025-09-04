import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "n_values,seeds",
    [
        ([1], [123]),
        pytest.param([2, 3], [123, 456], marks=pytest.mark.slow),
    ],
)
@pytest.mark.slow
def test_run_matrix_generates_outputs(
    tmp_path: Path, n_values: list[int], seeds: list[int]
) -> None:
    """Matrix run writes metrics for each combination."""

    script = Path(__file__).resolve().parents[2] / "scripts" / "eval_bench.py"
    outdir = tmp_path / "matrix_runs"
    cmd = [
        sys.executable,
        str(script),
        "+run_matrix=true",
        "preset=baselines/core",
        "+suites=[episodic]",
        f"+n_values={n_values}",
        f"+seeds={seeds}",
        f"outdir={outdir}",
    ]
    subprocess.run(cmd, check=True)

    metrics_paths = list((outdir / "episodic").glob("n*_seed*/metrics.json"))
    expected = {(n, seed) for n in n_values for seed in seeds}
    found = set()
    for path in metrics_paths:
        with path.open() as f:
            metrics = json.load(f)
        n = metrics["n"]
        seed = metrics["seed"]
        assert path.parent.name == f"n{n}_seed{seed}"
        found.add((n, seed))
    assert found == expected
