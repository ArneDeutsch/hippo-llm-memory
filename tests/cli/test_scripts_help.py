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
        assert "usage" in proc.stdout.lower()
        assert proc.returncode == 0
