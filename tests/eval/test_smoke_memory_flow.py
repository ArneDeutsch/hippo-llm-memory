import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_smoke_memory_flow() -> None:
    repo = Path(__file__).resolve().parents[2]
    run_id = "smoke_test"
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["MODEL"] = str(repo / "models" / "tiny-gpt2")
    base_run = repo / "runs" / run_id
    stores = base_run / "stores"
    if base_run.exists():
        shutil.rmtree(base_run, ignore_errors=True)
    baselines = base_run / "baselines"
    baselines.mkdir(parents=True, exist_ok=True)
    (baselines / "metrics.csv").write_text("suite,em_raw,em_norm,f1\n")
    stores.mkdir(parents=True, exist_ok=True)

    def build_dataset(suite: str, size: int) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hippo_eval.datasets.cli",
                "--suite",
                suite,
                "--size",
                str(size),
                "--seed",
                "1337",
                "--out",
                str(repo / "datasets" / suite),
            ],
            check=True,
            env=env,
            cwd=repo,
        )

    def run_eval(suite: str, preset: str, n: int, mode: str, session_id: str) -> None:
        cmd = [
            sys.executable,
            "scripts/eval_model.py",
            f"suite={suite}",
            f"preset={preset}",
            f"run_id={run_id}",
            f"n={n}",
            "seed=1337",
            f"mode={mode}",
            f"store_dir={stores}",
            f"session_id={session_id}",
            "compute.pre_metrics=true",
            f"model={env['MODEL']}",
        ]
        if mode == "teach":
            cmd.extend(["persist=true", "gating_enabled=false"])
        subprocess.run(cmd, check=True, env=env, cwd=repo)

    # Episodic teach + test
    build_dataset("episodic_cross_mem", 8)
    epi_session = f"hei_{run_id}"
    run_eval("episodic_cross_mem", "memory/hei_nw", 8, "teach", epi_session)
    run_eval("episodic_cross_mem", "memory/hei_nw", 8, "test", epi_session)
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_store.py",
            f"--run_id={run_id}",
            "--algo=hei_nw",
            "--kind=episodic",
            "--expect-nonzero-ratio=0.85",
        ],
        check=True,
        env=env,
        cwd=repo,
    )

    # Semantic teach + validate
    build_dataset("semantic_mem", 50)
    sem_session = f"sgc_{run_id}"
    run_eval("semantic_mem", "memory/sgc_rss", 50, "teach", sem_session)
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_store.py",
            f"--run_id={run_id}",
            "--algo=sgc_rss",
            "--kind=kg",
            "--expect-nodes=20",
            "--expect-edges=20",
        ],
        check=True,
        env=env,
        cwd=repo,
    )

    # Spatial teach + validate
    build_dataset("spatial_multi", 8)
    sp_session = f"smpd_{run_id}"
    run_eval("spatial_multi", "memory/smpd", 8, "teach", sp_session)
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_store.py",
            f"--run_id={run_id}",
            "--algo=smpd",
            "--kind=spatial",
        ],
        check=True,
        env=env,
        cwd=repo,
    )

    assert not list(base_run.glob("**/failed_preflight.json"))
    metrics_path = base_run / "hei_nw" / "episodic_cross_mem" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["retrieval"]["episodic"]["requests"] > 0
