import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PRESETS = ["baselines/core", "baselines/rag", "baselines/longctx"]


def test_baseline_presets_create_metrics(tmp_path: Path) -> None:
    """Each baseline preset writes metrics to the default runs directory."""

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
        assert (outdir / "metrics.json").exists()
