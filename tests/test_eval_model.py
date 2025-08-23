"""Smoke test for :mod:`scripts.eval_model`."""

import json
import subprocess
import sys
from pathlib import Path


def test_eval_model_dry_run(tmp_path: Path) -> None:
    """Run the evaluation harness and verify outputs and metadata."""

    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        "replay.cycles=1",
        f"outdir={outdir}",
        "+ablate.memory.episodic.hopfield=false",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)

    for name in ["metrics.json", "metrics.csv", "meta.json"]:
        assert (outdir / name).exists()

    meta = json.loads((outdir / "meta.json").read_text())
    assert meta["preset"] == "memory/hei_nw"
    assert meta["replay_cycles"] == 1
    assert meta["ablate"]["memory.episodic.hopfield"] is False
    assert len(meta["config_hash"]) == 64

    metrics = json.loads((outdir / "metrics.json").read_text())
    compute = metrics["metrics"]["compute"]
    assert isinstance(compute["time_ms_per_100"], float)
    assert isinstance(compute["rss_mb"], float)
