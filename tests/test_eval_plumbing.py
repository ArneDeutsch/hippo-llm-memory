"""Smoke tests for the synthetic evaluation harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("suite", ["episodic", "semantic", "spatial"])
def test_eval_bench(tmp_path: Path, suite: str) -> None:
    """The evaluation harness writes expected metrics for each suite."""

    outdir = tmp_path / suite
    cmd = [
        sys.executable,
        "scripts/eval_bench.py",
        "--suite",
        suite,
        "--n",
        "3",
        "--seed",
        "0",
        "--preset",
        "baselines/core",
        "--outdir",
        str(outdir),
    ]
    subprocess.run(cmd, check=True)

    metrics_path = outdir / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert data["suite"] == suite
    assert data["n"] == 3
    # Echo model should achieve perfect exact match.
    assert data["metrics"][suite]["em"] == 1.0

    csv_path = outdir / "metrics.csv"
    meta_path = outdir / "meta.json"
    assert csv_path.exists() and meta_path.exists()
