# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import subprocess
import sys
from pathlib import Path

import pytest

from hippo_mem.testing import FAKE_MODEL_ID


def run_eval(args: list[str], outdir: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        *args,
        f"model={FAKE_MODEL_ID}",
        f"outdir={outdir}",
        "n=1",
        "seed=1337",
        "mode=teach",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    assert (outdir / "meta.json").exists()


@pytest.mark.slow
def test_memory_override_shortcuts(tmp_path: Path) -> None:
    run_eval(
        [
            "suite=episodic_cross_mem",
            "preset=memory/hei_nw",
            "episodic.gate.tau=0.3",
        ],
        tmp_path / "epi",
    )

    run_eval(
        [
            "suite=semantic_mem",
            "preset=memory/sgc_rss",
            "relational.gate.threshold=0.4",
        ],
        tmp_path / "rel",
    )

    run_eval(
        [
            "suite=spatial_multi",
            "preset=memory/smpd",
            "spatial.gate.block_threshold=2.0",
        ],
        tmp_path / "spat",
    )
