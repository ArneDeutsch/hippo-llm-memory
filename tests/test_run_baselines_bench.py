import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CASES = [
    (["baselines/core"], [1337]),
    (["baselines/core", "baselines/rag"], [1337, 2025]),
]


def test_baselines_matrix_includes_new_suites() -> None:
    """SUITES constant lists all episodic variants."""

    from hippo_eval.eval import baselines

    assert baselines.SUITES == [
        "episodic",
        "semantic",
        "spatial",
        "episodic_multi",
        "episodic_cross",
        "episodic_capacity",
    ]


@pytest.mark.slow
@pytest.mark.parametrize("presets,seeds", CASES)
def test_run_baselines_accepts_run_id(presets: list[str], seeds: list[int]) -> None:
    """``run_baselines_bench.py`` emits metrics/meta for all combinations."""

    repo_root = Path(__file__).resolve().parents[1]
    run_id = "20250101"
    cmd = [
        sys.executable,
        "-m",
        "hippo_eval.eval.baselines",
        "--run-id",
        run_id,
        "--presets",
        *presets,
        "--suites",
        "episodic",
        "--sizes",
        "50",
        "--seeds",
        *(str(s) for s in seeds),
    ]
    env = {**os.environ, "ALLOW_BENCH": "1"}
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)
    base = repo_root / "runs" / run_id
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
            assert isinstance(compute["total_tokens"], int)

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
