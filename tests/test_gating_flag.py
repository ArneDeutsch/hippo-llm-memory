import csv
import json
import subprocess
import sys
from pathlib import Path


def test_gating_flag_disables_gate(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=semantic",
        "preset=memory/sgc_rss",
        "n=1",
        "seed=1337",
        f"outdir={outdir}",
        "gating_enabled=false",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    meta = json.loads((outdir / "meta.json").read_text())
    assert meta["gating_enabled"] is False
    assert meta["config"]["relational"]["gate"]["enabled"] is False
    with (outdir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["gating_enabled"] == "False"
