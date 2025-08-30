import subprocess
import sys
from pathlib import Path


def run_eval(args: list[str], outdir: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        *args,
        "model=models/tiny-gpt2",
        f"outdir={outdir}",
        "n=2",
        "seed=1337",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    assert (outdir / "meta.json").exists()


def test_memory_override_shortcuts(tmp_path: Path) -> None:
    run_eval(
        [
            "suite=episodic",
            "preset=memory/hei_nw",
            "episodic.gate.tau=0.3",
        ],
        tmp_path / "epi",
    )

    run_eval(
        [
            "suite=semantic",
            "preset=memory/sgc_rss",
            "relational.gate.threshold=0.4",
        ],
        tmp_path / "rel",
    )

    run_eval(
        [
            "suite=spatial",
            "preset=memory/smpd",
            "spatial.gate.block_threshold=2.0",
        ],
        tmp_path / "spat",
    )
