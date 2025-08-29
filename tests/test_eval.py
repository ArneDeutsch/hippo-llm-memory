"""Integration tests for the evaluation harness presets.

The tests in this module execute the lightweight evaluation script with the
baseline presets to ensure that metrics and metadata files are produced with
the expected structure.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

PRESETS = [
    "baselines/core",
    pytest.param("baselines/rag", marks=pytest.mark.slow),
    pytest.param("baselines/longctx", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("preset", PRESETS)
def test_baseline_presets_create_metrics(tmp_path: Path, preset: str) -> None:
    """Each baseline preset writes well-formed metrics and metadata files."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_bench.py"
    cmd = [
        sys.executable,
        str(script),
        "suite=episodic",
        f"preset={preset}",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True, cwd=tmp_path)
    # discover the date directory created by the run
    date_dir = max((tmp_path / "runs").iterdir(), key=lambda p: p.name)
    outdir = date_dir / "baselines" / Path(preset).name / "episodic"

    metrics_path = outdir / "metrics.json"
    meta_path = outdir / "meta.json"
    csv_path = outdir / "metrics.csv"

    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert data["preset"] == preset
    assert data["suite"] == "episodic"
    assert isinstance(data["metrics"]["episodic"]["em_raw"], float)
    compute = data["metrics"]["compute"]
    assert isinstance(compute["total_tokens"], int)
    assert isinstance(compute["input_tokens"], int)
    assert isinstance(compute["generated_tokens"], int)
    assert isinstance(compute["time_ms_per_100"], float)
    assert isinstance(compute["rss_mb"], float)
    assert compute["latency_ms_mean"] > 0

    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    model_meta = meta.get("model")
    if isinstance(model_meta, dict):
        assert isinstance(model_meta["id"], str)
        assert isinstance(model_meta["chat_template_used"], bool)
    else:
        assert isinstance(model_meta, str)
    assert meta.get("seed") == 0
    assert meta.get("suite") == "episodic"
    assert meta.get("preset") == preset
    assert meta.get("n") == data["n"]
    assert isinstance(meta.get("python"), str)
    assert isinstance(meta.get("platform"), str)
    assert len(meta.get("pip_hash", "")) == 64
    assert isinstance(meta.get("cpu"), str)

    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows and float(rows[0]["latency_ms"]) > 0
