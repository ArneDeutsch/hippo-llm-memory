import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hippo_eval.eval import harness


def _run_eval(tmp_path: Path, *overrides: str):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "eval_bench.py"
    cmd = [
        sys.executable,
        str(script),
        "suite=episodic",
        "preset=baselines/core",
        "dry_run=true",
        "n=1",
        *overrides,
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    run_id = None
    for arg in overrides:
        if arg.startswith("run_id="):
            run_id = arg.split("=", 1)[1]
    if run_id:
        src = repo_root / "runs" / run_id
        dst = tmp_path / "runs" / run_id
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dst)
    return result


def test_paths_use_run_id(tmp_path: Path) -> None:
    run_id = "RID123"
    _run_eval(tmp_path, f"run_id={run_id}")
    outdir = tmp_path / "runs" / run_id / "baselines" / "core" / "episodic"
    assert (outdir / "metrics.json").exists()


def test_invalid_run_id_rejected() -> None:
    cfg = OmegaConf.create({"run_matrix": True, "presets": ["foo"], "run_id": "bad id"})
    with pytest.raises(ValueError, match="Invalid RUN_ID"):
        harness.main(cfg)
