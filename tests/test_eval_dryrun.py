import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Exercise baseline presets across suites in dry‑run mode. Only one
# representative combination runs in the default suite; the rest are
# marked as slow to keep CI fast while preserving coverage when the
# ``slow`` suite is executed.
CASES = [
    ("baselines/core", "episodic"),
    pytest.param("baselines/rag", "episodic", marks=pytest.mark.slow),
    pytest.param("baselines/longctx", "episodic", marks=pytest.mark.slow),
    pytest.param("baselines/core", "semantic", marks=pytest.mark.slow),
    pytest.param("baselines/core", "spatial", marks=pytest.mark.slow),
    pytest.param("baselines/rag", "semantic", marks=pytest.mark.slow),
    pytest.param("baselines/rag", "spatial", marks=pytest.mark.slow),
    pytest.param("baselines/longctx", "semantic", marks=pytest.mark.slow),
    pytest.param("baselines/longctx", "spatial", marks=pytest.mark.slow),
]


@pytest.mark.slow
@pytest.mark.parametrize("preset,suite", CASES)
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

    metrics = json.loads((outdir / "metrics.json").read_text())
    meta = json.loads((outdir / "meta.json").read_text())

    assert metrics["preset"] == preset
    assert metrics["suite"] == suite
    compute = metrics["metrics"]["compute"]
    assert compute["time_ms_per_100"] > 0
    assert compute["latency_ms_mean"] > 0
    suite_metrics = metrics["metrics"][suite]
    assert "em_raw" in suite_metrics
    assert "memory_hit_rate" in suite_metrics
    assert "latency_ms_delta" in suite_metrics
    with (outdir / "metrics.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "memory_hit" in reader.fieldnames
        assert "retrieval_latency_ms" in reader.fieldnames

    assert meta["preset"] == preset
    assert meta["suite"] == suite
