import subprocess
import sys
from pathlib import Path

PRESETS = [
    "baselines/core",
    "baselines/rag",
    "baselines/longctx",
    "memory/hei_nw",
    "memory/sgc_rss",
    "memory/smpd",
    "memory/all",
]

SUITES = ["episodic", "semantic", "spatial"]


def test_dry_run_smoke(tmp_path: Path) -> None:
    """Dry run the harness across all suites and presets."""

    for preset in PRESETS:
        for suite in SUITES:
            outdir = tmp_path / preset.replace("/", "_") / suite
            cmd = [
                sys.executable,
                "scripts/eval_bench.py",
                f"suite={suite}",
                f"preset={preset}",
                "dry_run=true",
                f"outdir={outdir}",
            ]
            subprocess.run(cmd, check=True)
            for name in ["metrics.json", "metrics.csv", "meta.json"]:
                assert (outdir / name).exists()
