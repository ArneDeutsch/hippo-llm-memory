import shutil
import subprocess
import sys
from pathlib import Path


def test_run_baselines_accepts_date() -> None:
    """``run_baselines.py`` propagates ``--date`` to ``eval_bench``."""

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_baselines.py"
    date = "20250101"
    cmd = [
        sys.executable,
        str(script),
        "--date",
        date,
        "--presets",
        "baselines/core",
        "--suites",
        "episodic",
        "--sizes",
        "50",
        "--seeds",
        "1337",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)
    out = repo_root / "runs" / date / "baselines/core" / "episodic" / "50_1337"
    try:
        assert (out / "metrics.json").exists()
    finally:
        if out.parent.parent.parent.exists():
            shutil.rmtree(repo_root / "runs" / date)
