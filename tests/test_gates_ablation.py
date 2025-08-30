import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_run_memory_gate_flags(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/run_memory.py",
        "--suite=semantic",
        "--preset=memory/sgc_rss",
        f"--outdir={outdir}",
        "--n=1",
        "--seed=1337",
        "--mode=teach",
        "--model=models/tiny-gpt2",
        "--dry-run",
        "--relational-gate=off",
    ]
    subprocess.run(cmd, check=True)
    meta = json.loads((outdir / "meta.json").read_text())
    assert meta["gating_enabled"] is False
