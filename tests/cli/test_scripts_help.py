import subprocess
import sys

SCRIPTS = [
    ["-m", "hippo_eval.bench", "--help"],
    ["-m", "hippo_eval.datasets.cli", "--help"],
]


def test_cli_help() -> None:
    for args in SCRIPTS:
        subprocess.run(
            [sys.executable, *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
