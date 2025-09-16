# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hippo_mem.testing import FAKE_MODEL_ID


def _run(primary: str, outdir: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic_cross_mem",
        "preset=baselines/core",
        "n=1",
        "seed=1337",
        f"model={FAKE_MODEL_ID}",
        f"outdir={outdir}",
        "dry_run=true",
        "mode=teach",
        f"primary_em={primary}",
    ]
    subprocess.run(cmd, check=True)
    return json.loads((outdir / "metrics.json").read_text())


@pytest.mark.slow
def test_primary_em_switch(tmp_path: Path) -> None:
    metrics_norm = _run("norm", tmp_path / "norm")
    suite_norm = metrics_norm["metrics"]["episodic_cross_mem"]
    assert suite_norm["pre_em"] == suite_norm["pre_em_norm"]
    assert "pre_em_raw" in suite_norm and "pre_em_norm" in suite_norm

    metrics_raw = _run("raw", tmp_path / "raw")
    suite_raw = metrics_raw["metrics"]["episodic_cross_mem"]
    assert suite_raw["pre_em"] == suite_raw["pre_em_raw"]
    assert "pre_em_raw" in suite_raw and "pre_em_norm" in suite_raw
