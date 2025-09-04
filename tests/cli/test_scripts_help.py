import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "eval_model.py",
    "datasets_cli.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_help(script: str) -> None:
    cmd = [sys.executable, f"scripts/{script}", "--help"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT, env=env
    )
