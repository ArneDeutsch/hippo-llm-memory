import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_smoke_memory_flow(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    run_id = "smoke_test"
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["MODEL"] = str(repo / "models" / "tiny-gpt2")
    base_run = repo / "runs" / run_id
    stores = base_run / "stores"
    baselines = base_run / "baselines"
    baselines.mkdir(parents=True, exist_ok=True)
    (baselines / "metrics.csv").write_text("suite,em_raw,em_norm,f1\n")
    stores.mkdir(parents=True, exist_ok=True)
    session_id = f"hei_{run_id}"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "hippo_eval.datasets.cli",
            "--suite",
            "episodic_cross_mem",
            "--size",
            "5",
            "--seed",
            "1337",
            "--out",
            str(repo / "datasets" / "episodic_cross_mem"),
        ],
        check=True,
        env=env,
        cwd=repo,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/eval_model.py",
            "suite=episodic_cross_mem",
            "preset=memory/hei_nw",
            f"run_id={run_id}",
            "n=5",
            "seed=1337",
            "mode=teach",
            "persist=true",
            f"store_dir={stores}",
            f"session_id={session_id}",
            "compute.pre_metrics=true",
            f"model={env['MODEL']}",
        ],
        check=True,
        env=env,
        cwd=repo,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/eval_model.py",
            "suite=episodic_cross_mem",
            "preset=memory/hei_nw",
            f"run_id={run_id}",
            "n=5",
            "seed=1337",
            "mode=test",
            f"store_dir={stores}",
            f"session_id={session_id}",
            "compute.pre_metrics=true",
            f"model={env['MODEL']}",
        ],
        check=True,
        env=env,
        cwd=repo,
    )

    assert not list(base_run.glob("**/failed_preflight.json"))
    store_file = stores / "hei_nw" / session_id / "episodic.jsonl"
    assert store_file.exists()
    metrics_files = list(base_run.glob("**/metrics.json"))
    assert metrics_files
    json.loads(metrics_files[0].read_text())
