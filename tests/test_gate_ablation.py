import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "preset,suite,key",
    [
        ("memory/hei_nw", "episodic", "episodic.use_gate"),
        ("memory/sgc_rss", "semantic", "relational.gate.enabled"),
        ("memory/smpd", "spatial", "spatial.gate.enabled"),
    ],
)
def test_gate_ablation(tmp_path: Path, preset: str, suite: str, key: str) -> None:
    outdir = tmp_path / preset.replace("/", "_")
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        f"suite={suite}",
        f"preset={preset}",
        "n=1",
        "seed=1337",
        f"outdir={outdir}",
        f"+ablate.{key}=false",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    meta = json.loads((outdir / "meta.json").read_text())
    assert meta["ablate"][key] is False
    with (outdir / "metrics.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["gating_enabled"] == "False"
