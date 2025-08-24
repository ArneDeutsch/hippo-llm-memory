import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CASES = [
    (["baselines/core"], [1337]),
    pytest.param(["baselines/core", "baselines/rag"], [1337, 2025], marks=pytest.mark.slow),
]


@pytest.mark.parametrize("presets,seeds", CASES)
def test_run_baselines_accepts_date(presets: list[str], seeds: list[int]) -> None:
    """``run_baselines.py`` emits metrics/meta for all combinations."""

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_baselines.py"
    date = "20250101"
    cmd = [
        sys.executable,
        str(script),
        "--date",
        date,
        "--presets",
        *presets,
        "--suites",
        "episodic",
        "--sizes",
        "50",
        "--seeds",
        *(str(s) for s in seeds),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)
    base = repo_root / "runs" / date
    try:
        for preset, seed in itertools.product(presets, seeds):
            out = base / preset / "episodic" / f"50_{seed}"
            metrics = json.loads((out / "metrics.json").read_text())
            meta = json.loads((out / "meta.json").read_text())

            # metrics.json invariants
            assert metrics["suite"] == "episodic"
            assert metrics["preset"] == preset
            assert metrics["n"] == 50
            assert metrics["seed"] == seed
            compute = metrics["metrics"]["compute"]
            assert isinstance(compute["tokens"], int)

            # meta.json required fields
            assert meta["suite"] == "episodic"
            assert meta["preset"] == preset
            assert meta["n"] == 50
            assert meta["seed"] == seed
            assert len(meta["git_sha"]) == 40
            assert len(meta["config_hash"]) == 64
            for field in ["python", "platform", "pip_hash"]:
                assert field in meta

        # run-count invariant
        runs = list(base.rglob("metrics.json"))
        assert len(runs) == len(presets) * len(seeds)
    finally:
        if base.exists():
            shutil.rmtree(base)
