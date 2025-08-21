import csv
import subprocess
import sys
from pathlib import Path


def test_post_replay_cycle_generates_metrics(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/eval_bench.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=2",
        "seed=0",
        "post_replay_cycles=1",
        f"outdir={outdir}",
    ]
    subprocess.run(cmd, check=True)
    csv_path = outdir / "metrics.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        flags = [row["flags"] for row in csv.DictReader(f)]
    assert "pre_replay" in flags
    assert "post_replay_1" in flags
