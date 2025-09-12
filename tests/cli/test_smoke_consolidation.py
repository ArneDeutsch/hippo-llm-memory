# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_smoke_consolidation_tiny(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "hippo_eval.consolidation.eval",
        "--phase",
        "pre",
        "--suite",
        "episodic",
        "--n",
        "1",
        "--seed",
        "0",
        "--outdir",
        str(outdir),
        "--allow-tiny-test-model",
    ]
    subprocess.run(cmd, check=True, env=env, cwd=repo)
    assert (outdir / "metrics.json").exists()
