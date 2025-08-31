import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_eval_model_run_matrix(tmp_path: Path) -> None:
    """Matrix run writes metrics for each combination."""

    outdir = tmp_path / "matrix"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "+run_matrix=true",
        "preset=memory/hei_nw",
        "+suites=[episodic]",
        "+n_values=[2]",
        "+seeds=[1337]",
        "model=models/tiny-gpt2",
        f"outdir={outdir}",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    expected = outdir / "episodic" / "2_1337" / "metrics.json"
    assert expected.exists()
    metrics = json.loads(expected.read_text())
    assert metrics["n"] == 2
    assert metrics["seed"] == 1337


@pytest.mark.slow
def test_eval_model_run_matrix_episodic_multi(tmp_path: Path) -> None:
    """Matrix run resolves datasets in subdirectories."""

    outdir = tmp_path / "matrix"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "+run_matrix=true",
        "preset=memory/hei_nw",
        "tasks=[episodic_multi]",
        "n_values=[2]",
        "seeds=[1337]",
        "model=models/tiny-gpt2",
        f"outdir={outdir}",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    expected = outdir / "episodic_multi" / "2_1337" / "metrics.json"
    assert expected.exists()


@pytest.mark.slow
def test_eval_model_run_matrix_date(tmp_path: Path) -> None:
    """Matrix run handles numeric date without explicit outdir."""

    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "data").symlink_to(repo_root / "data", target_is_directory=True)
    (tmp_path / "models").symlink_to(repo_root / "models", target_is_directory=True)
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_model.py"),
        "+run_matrix=true",
        "preset=memory/hei_nw",
        "+suites=[episodic]",
        "+n_values=[2]",
        "+seeds=[1337]",
        "model=models/tiny-gpt2",
        "dry_run=true",
        "date=20250824",
    ]
    subprocess.run(cmd, check=True, cwd=tmp_path, env=env)
    expected = tmp_path / "runs" / "20250824" / "hei_nw" / "episodic" / "2_1337" / "metrics.json"
    assert expected.exists()


@pytest.mark.slow
def test_eval_model_run_matrix_baseline(tmp_path: Path) -> None:
    """Baseline matrix run uses local tiny model."""

    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "data").symlink_to(repo_root / "data", target_is_directory=True)
    (tmp_path / "models").symlink_to(repo_root / "models", target_is_directory=True)
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_model.py"),
        "+run_matrix=true",
        "preset=baselines/core",
        "+suites=[episodic]",
        "+n_values=[2]",
        "+seeds=[1337]",
        "model=models/tiny-gpt2",
        "dry_run=true",
        "date=20250824",
    ]
    subprocess.run(cmd, check=True, cwd=tmp_path, env=env)
    expected = (
        tmp_path
        / "runs"
        / "20250824"
        / "baselines"
        / "core"
        / "episodic"
        / "2_1337"
        / "metrics.json"
    )
    assert expected.exists()


@pytest.mark.slow
def test_eval_model_run_matrix_date_time(tmp_path: Path) -> None:
    """Matrix run handles timestamped date without explicit outdir."""

    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "data").symlink_to(repo_root / "data", target_is_directory=True)
    (tmp_path / "models").symlink_to(repo_root / "models", target_is_directory=True)
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_model.py"),
        "+run_matrix=true",
        "preset=memory/hei_nw",
        "+suites=[episodic]",
        "+n_values=[2]",
        "+seeds=[1337]",
        "model=models/tiny-gpt2",
        "dry_run=true",
        "date=20250829_0841",
    ]
    subprocess.run(cmd, check=True, cwd=tmp_path, env=env)
    expected = (
        tmp_path / "runs" / "20250829_0841" / "hei_nw" / "episodic" / "2_1337" / "metrics.json"
    )
    assert expected.exists()


@pytest.mark.slow
def test_eval_model_run_matrix_presets(tmp_path: Path) -> None:
    """Matrix run with multiple presets writes outputs for each preset."""

    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "+run_matrix=true",
        "presets=[baselines/core,baselines/longctx]",
        "tasks=[episodic]",
        "n_values=[2]",
        "seeds=[1337]",
        "model=models/tiny-gpt2",
        f"outdir={tmp_path}",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    for preset in ("baselines/core", "baselines/longctx"):
        metrics = tmp_path / preset / "episodic" / "2_1337" / "metrics.json"
        assert metrics.exists()
