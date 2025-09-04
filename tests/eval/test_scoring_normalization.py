import json
import subprocess
import sys
from pathlib import Path

import pytest

from hippo_eval.eval.score import em_norm, normalize


def test_normalize_and_em() -> None:
    assert normalize("The, apple!") == "apple"
    assert em_norm("An apple", "apple") == 0
    assert em_norm("apple", "banana") == 0


@pytest.mark.slow
def test_harness_writes_em_scores(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=1",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={outdir}",
        "mode=teach",
        "dry_run=true",
    ]
    subprocess.run(cmd, check=True)
    metrics = json.loads((outdir / "metrics.json").read_text())
    suite = metrics["metrics"]["episodic"]
    assert "pre_em_raw" in suite and "pre_em_norm" in suite
