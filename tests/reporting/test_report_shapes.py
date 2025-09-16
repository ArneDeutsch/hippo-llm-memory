# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hippo_mem.testing import FAKE_MODEL_ID


@pytest.mark.slow
def test_metrics_csv_has_diagnostics(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic_cross_mem",
        "preset=baselines/core",
        "n=1",
        "seed=1337",
        f"model={FAKE_MODEL_ID}",
        f"outdir={outdir}",
        "mode=teach",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)

    csv_path = outdir / "metrics.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    for col in ["pred_len", "gold_len", "overlong", "format_violation"]:
        assert col in headers

    metrics = json.loads((outdir / "metrics.json").read_text())
    diag = metrics["diagnostics"]["episodic_cross_mem"]
    assert "pre_overlong" in diag
    assert "pre_format_violation" in diag
