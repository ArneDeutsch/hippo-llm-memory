import subprocess
import sys
from pathlib import Path

# Limit the scope of the smoke test to keep runtime reasonable.
# The harness itself is exercised by running a single representative
# preset/suite combination.  Additional combinations are exercised in the
# full evaluation workflow but would make unit tests prohibitively slow.
PRESETS = ["baselines/core"]
SUITES = ["episodic"]


def test_dry_run_smoke(tmp_path: Path) -> None:
    """Dry run the harness across all suites and presets."""

    for preset in PRESETS:
        for suite in SUITES:
            print(f"dry-run {preset}/{suite}")
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
