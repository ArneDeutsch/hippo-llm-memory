"""Ensure replay cycles accept flat and nested keys."""

import json
import subprocess
import sys
from pathlib import Path


def test_replay_cycles_config_fallback(tmp_path: Path) -> None:
    out_nested = tmp_path / "nested"
    cmd_nested = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic_cross_mem",
        "preset=memory/hei_nw",
        "n=1",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={out_nested}",
        "replay.cycles=1",
        "dry_run=true",
    ]
    subprocess.run(cmd_nested, check=True)
    meta_nested = json.loads((out_nested / "meta.json").read_text())
    metrics_nested = json.loads((out_nested / "metrics.json").read_text())

    out_flat = tmp_path / "flat"
    cmd_flat = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic_cross_mem",
        "preset=memory/hei_nw",
        "n=1",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={out_flat}",
        "replay_cycles=1",
        "dry_run=true",
    ]
    subprocess.run(cmd_flat, check=True)
    meta_flat = json.loads((out_flat / "meta.json").read_text())
    metrics_flat = json.loads((out_flat / "metrics.json").read_text())

    assert meta_nested["replay_cycles"] == 1
    assert meta_flat["replay_cycles"] == 1
    assert metrics_nested["replay"]["samples"] == metrics_flat["replay"]["samples"] == 1
