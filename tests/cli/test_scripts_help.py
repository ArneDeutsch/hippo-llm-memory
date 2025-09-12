# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import subprocess
import sys

SCRIPTS = [
    ["-m", "hippo_eval.bench", "--help"],
    ["-m", "hippo_eval.datasets.cli", "--help"],
]


def test_cli_help() -> None:
    for args in SCRIPTS:
        proc = subprocess.run(
            [sys.executable, *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lower = proc.stdout.lower()
        assert "usage" in lower or "hydra" in lower
        assert proc.returncode == 0
