import subprocess
import sys
from pathlib import Path

import pytest

# Exercise every baseline preset across all suites in dry‑run mode.
# Each run evaluates a single example to keep runtime reasonable while
# confirming that the harness loads the corresponding configuration.
PRESETS = ["baselines/core", "baselines/rag", "baselines/longctx"]
SUITES = ["episodic", "semantic", "spatial"]


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("suite", SUITES)
def test_dry_run_smoke(tmp_path: Path, preset: str, suite: str) -> None:
    """Dry run the harness for a single preset/suite combination."""

    print(f"dry-run {preset}/{suite}")
    outdir = tmp_path / preset.replace("/", "_") / suite
    cmd = [
        sys.executable,
        "scripts/eval_bench.py",
        f"suite={suite}",
        f"preset={preset}",
        "dry_run=true",
        "n=1",
        f"outdir={outdir}",
    ]
    subprocess.run(cmd, check=True)
    for name in ["metrics.json", "metrics.csv", "meta.json"]:
        assert (outdir / name).exists()
