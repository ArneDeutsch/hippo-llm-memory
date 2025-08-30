import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_metrics_include_post_and_delta(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    store_dir = tmp_path / "stores"
    env = dict(os.environ)
    env["HF_MODEL_PATH"] = str(Path("models/tiny-gpt2"))
    cmd_teach = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        f"outdir={outdir}",
        f"store_dir={store_dir}",
        "session_id=s1",
        "mode=teach",
        "persist=true",
        "dry_run=true",
    ]
    subprocess.run(cmd_teach, check=True, env=env)

    cmd_test = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        f"outdir={outdir}",
        f"store_dir={store_dir}",
        "session_id=s1",
        "mode=test",
        "persist=true",
        "dry_run=true",
    ]
    subprocess.run(cmd_test, check=True, env=env)

    cmd_replay = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        f"outdir={outdir}",
        f"store_dir={store_dir}",
        "session_id=s1",
        "mode=replay",
        "persist=true",
        "dry_run=true",
    ]
    subprocess.run(cmd_replay, check=True, env=env)

    metrics_path = outdir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    suite_metrics = metrics["metrics"]["episodic"]
    assert "pre_em" in suite_metrics
    assert "post_em" in suite_metrics
    assert "delta_em" in suite_metrics
    assert suite_metrics["delta_em"] == pytest.approx(
        suite_metrics["post_em"] - suite_metrics["pre_em"]
    )
