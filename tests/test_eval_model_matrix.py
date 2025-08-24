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
        f"outdir={outdir}",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    expected = outdir / "episodic" / "2_1337" / "metrics.json"
    assert expected.exists()
    metrics = json.loads(expected.read_text())
    assert metrics["n"] == 2
    assert metrics["seed"] == 1337


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
