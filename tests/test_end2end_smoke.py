from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, *cmd], check=True, cwd=cwd)


def test_end_to_end_smoke(tmp_path: Path) -> None:
    """Run baseline and memory presets and validate metrics, stores and gating."""

    repo = Path(__file__).resolve().parent.parent

    baseline_out = tmp_path / "baseline"
    _run(
        [
            str(repo / "scripts" / "eval_model.py"),
            "suite=episodic",
            "preset=baselines/core",
            "n=5",
            "seed=1337",
            "model=models/tiny-gpt2",
            f"outdir={baseline_out}",
            "dry_run=true",
        ],
        repo,
    )
    baseline_metrics = json.loads((baseline_out / "metrics.json").read_text())
    pre_em = baseline_metrics["metrics"]["episodic"]["pre_em"]
    assert pre_em == pre_em  # no NaN

    store_dir = tmp_path / "stores"
    teach_out = tmp_path / "teach"
    _run(
        [
            str(repo / "scripts" / "eval_model.py"),
            "suite=episodic",
            "preset=memory/hei_nw",
            "n=5",
            "seed=1337",
            "model=models/tiny-gpt2",
            f"outdir={teach_out}",
            f"store_dir={store_dir}",
            "session_id=s1",
            "mode=teach",
            "persist=true",
            "dry_run=true",
        ],
        repo,
    )

    mem_out = tmp_path / "memory"
    _run(
        [
            str(repo / "scripts" / "eval_model.py"),
            "suite=episodic",
            "preset=memory/hei_nw",
            "n=5",
            "seed=1337",
            "model=models/tiny-gpt2",
            f"outdir={mem_out}",
            f"store_dir={store_dir}",
            "session_id=s1",
            "replay.cycles=1",
            "persist=true",
            "dry_run=true",
        ],
        repo,
    )
    mem_metrics = json.loads((mem_out / "metrics.json").read_text())
    pre = mem_metrics["metrics"]["episodic"]["pre_em"]
    assert pre == pre  # no NaN
    assert mem_metrics["replay"]["samples"] >= 1
    gates = mem_metrics.get("gates")
    if gates is not None:
        assert gates.get("episodic", {}).get("attempts", 0) > 0

    store_file = store_dir / "hei_nw" / "s1" / "episodic.jsonl"
    assert store_file.exists()
    assert '"provenance": "dummy"' not in store_file.read_text()
