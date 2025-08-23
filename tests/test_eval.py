"""Integration tests for the evaluation harness presets.

The tests in this module execute the lightweight evaluation script with the
baseline presets to ensure that metrics and metadata files are produced with
the expected structure.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PRESETS = ["baselines/core", "baselines/rag", "baselines/longctx"]


def test_baseline_presets_create_metrics(tmp_path: Path) -> None:
    """Each baseline preset writes well-formed metrics and metadata files."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_bench.py"
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    for preset in PRESETS:
        cmd = [
            sys.executable,
            str(script),
            "suite=episodic",
            f"preset={preset}",
            "dry_run=true",
        ]
        subprocess.run(cmd, check=True, cwd=tmp_path)
        outdir = tmp_path / "runs" / date / "baselines" / Path(preset).name / "episodic"

        metrics_path = outdir / "metrics.json"
        meta_path = outdir / "meta.json"

        assert metrics_path.exists()
        data = json.loads(metrics_path.read_text())
        assert data["preset"] == preset
        assert data["suite"] == "episodic"
        assert isinstance(data["metrics"]["episodic"]["em"], float)
        compute = data["metrics"]["compute"]
        assert isinstance(compute["tokens"], int)
        assert isinstance(compute["time_ms_per_100"], float)
        assert isinstance(compute["rss_mb"], float)

        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert isinstance(meta.get("model"), str)
        assert meta.get("seed") == 0
        assert isinstance(meta.get("python"), str)
        assert isinstance(meta.get("platform"), str)
        assert len(meta.get("pip_hash", "")) == 64
        assert isinstance(meta.get("cpu"), str)
