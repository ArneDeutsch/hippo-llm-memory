"""Smoke tests for the synthetic evaluation harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts import eval_bench


@pytest.mark.parametrize("suite", ["episodic", "semantic", "spatial"])
def test_eval_bench(tmp_path: Path, suite: str) -> None:
    """The evaluation harness writes expected metrics for each suite."""

    outdir = tmp_path / suite
    cmd = [
        sys.executable,
        "scripts/eval_bench.py",
        f"suite={suite}",
        "preset=baselines/core",
        "n=3",
        "seed=0",
        f"outdir={outdir}",
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


def test_ablate_disables_hopfield(tmp_path: Path) -> None:
    """Ablation flags propagate to module configs and metadata."""

    outdir = tmp_path / "abl"
    cmd = [
        sys.executable,
        "scripts/eval_bench.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=0",
        f"outdir={outdir}",
        "+ablate.memory.episodic.hopfield=false",
    ]
    subprocess.run(cmd, check=True)

    meta = json.loads((outdir / "meta.json").read_text())
    assert meta["ablate"]["memory.episodic.hopfield"] is False

    modules = eval_bench._init_modules("hei_nw", meta["ablate"])
    assert modules["episodic"]["store"].config["hopfield"] is False
