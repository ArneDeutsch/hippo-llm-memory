"""Smoke test for :mod:`scripts.eval_model`."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


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
        "model=models/tiny-gpt2",
        "replay.cycles=1",
        f"outdir={outdir}",
        "+ablate.memory.episodic.hopfield=false",
        "gating_enabled=false",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)

    for name in ["metrics.json", "metrics.csv", "meta.json"]:
        assert (outdir / name).exists()

    meta = json.loads((outdir / "meta.json").read_text())
    assert meta["suite"] == "episodic"
    assert meta["preset"] == "memory/hei_nw"
    assert meta["n"] == 2
    assert meta["replay_cycles"] == 1
    assert meta["mode"] == "test"
    assert meta["persist"] is False
    assert meta["ablate"]["memory.episodic.hopfield"] is False
    assert len(meta["config_hash"]) == 64
    assert isinstance(meta["model"], dict)
    assert isinstance(meta["model"]["chat_template_used"], bool)

    metrics = json.loads((outdir / "metrics.json").read_text())
    compute = metrics["metrics"]["compute"]
    assert isinstance(compute["time_ms_per_100"], float)
    assert isinstance(compute["rss_mb"], float)
    assert compute["latency_ms_mean"] > 0
    assert metrics["replay"]["samples"] == 2
    assert metrics["store"]["size"] >= 1
    suite_metrics = metrics["metrics"]["episodic"]
    assert "pre_refusal_rate" in suite_metrics
    assert metrics["gates"]["relational"]["attempts"] == 0
    assert "accepts" in metrics["gates"]["relational"]

    with (outdir / "metrics.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows and float(rows[0]["latency_ms"]) > 0


def test_eval_model_cli_flags(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    store_dir = tmp_path / "stores"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={outdir}",
        f"store_dir={store_dir}",
        "session_id=abc",
        "mode=teach",
        "persist=true",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    meta = json.loads((outdir / "meta.json").read_text())
    assert meta["mode"] == "teach"
    assert meta["store_dir"] == str(store_dir / "hei_nw")
    assert meta["session_id"] == "abc"
    assert meta["persist"] is True


def test_teach_persists_and_skips_metrics(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    store_dir = tmp_path / "stores"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={outdir}",
        f"store_dir={store_dir}",
        "session_id=s1",
        "mode=teach",
        "persist=true",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)

    # store persisted
    assert (store_dir / "hei_nw" / "s1" / "episodic.jsonl").exists()

    # metrics should not include scores
    with (outdir / "metrics.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(row["em_raw"] == "" for row in rows)


def test_load_store_and_memory_off(tmp_path: Path) -> None:
    outdir_teach = tmp_path / "teach"
    store_dir = tmp_path / "stores"
    cmd_teach = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={outdir_teach}",
        f"store_dir={store_dir}",
        "session_id=s1",
        "mode=teach",
        "persist=true",
        "dry_run=true",
    ]
    subprocess.run(cmd_teach, check=True)

    # memory on, store loaded
    outdir_test = tmp_path / "test_on"
    cmd_test = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={outdir_test}",
        f"store_dir={store_dir}",
        "session_id=s1",
        "mode=test",
        "dry_run=true",
    ]
    subprocess.run(cmd_test, check=True)
    metrics = json.loads((outdir_test / "metrics.json").read_text())
    assert metrics["retrieval"]["episodic"]["requests"] >= 1

    # memory explicitly off
    outdir_off = tmp_path / "test_off"
    cmd_off = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={outdir_off}",
        f"store_dir={store_dir}",
        "session_id=s1",
        "mode=test",
        "dry_run=true",
        "memory_off=true",
    ]
    subprocess.run(cmd_off, check=True)
    metrics_off = json.loads((outdir_off / "metrics.json").read_text())
    assert metrics_off["retrieval"]["episodic"]["requests"] == 0


@pytest.mark.parametrize(
    "preset,expected",
    [
        ("baselines/core", ["baselines", "core"]),
        ("memory/hei_nw", ["hei_nw"]),
    ],
)
def test_date_parameter_controls_outdir(tmp_path: Path, preset: str, expected: list[str]) -> None:
    """CLI ``date`` parameter selects the run subdirectory."""

    repo_root = Path(__file__).resolve().parent.parent
    # provide data and model via symlinks so harness can resolve them
    (tmp_path / "data").symlink_to(repo_root / "data", target_is_directory=True)
    (tmp_path / "models").symlink_to(repo_root / "models", target_is_directory=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_model.py"),
        "suite=episodic",
        f"preset={preset}",
        "n=2",
        "seed=1337",
        "model=models/tiny-gpt2",
        "date=20250101_0101",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True, cwd=tmp_path)

    run_dir = tmp_path / "runs" / "20250101_0101"
    for part in expected:
        run_dir /= part
    run_dir /= "episodic"
    assert (run_dir / "meta.json").exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["date"] == "20250101_0101"
