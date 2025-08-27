import json
import subprocess
import sys
from pathlib import Path


def _run(primary: str, outdir: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=baselines/core",
        "n=1",
        "seed=1337",
        f"outdir={outdir}",
        "dry_run=true",
        f"primary_em={primary}",
    ]
    subprocess.run(cmd, check=True)
    return json.loads((outdir / "metrics.json").read_text())


def test_primary_em_switch(tmp_path: Path) -> None:
    metrics_norm = _run("norm", tmp_path / "norm")
    suite_norm = metrics_norm["metrics"]["episodic"]
    assert suite_norm["pre_em"] == suite_norm["pre_em_norm"]
    assert "pre_em_raw" in suite_norm and "pre_em_norm" in suite_norm

    metrics_raw = _run("raw", tmp_path / "raw")
    suite_raw = metrics_raw["metrics"]["episodic"]
    assert suite_raw["pre_em"] == suite_raw["pre_em_raw"]
    assert "pre_em_raw" in suite_raw and "pre_em_norm" in suite_raw
