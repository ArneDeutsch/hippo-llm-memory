import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_replay_writes_post_and_delta(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    store_dir = tmp_path / "stores"
    base_cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=1",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"outdir={outdir}",
        f"store_dir={store_dir}",
        "session_id=s1",
    ]
    subprocess.run(base_cmd + ["mode=teach", "persist=true"], check=True)
    baseline_path = outdir.parents[2] / "baselines" / "metrics.csv"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text("suite,em_raw,em_norm,f1\n")
    subprocess.run(base_cmd, check=True)
    subprocess.run(
        base_cmd + ["mode=replay", "persist=true", "replay.cycles=1"],
        check=True,
    )
    data = json.loads((outdir / "metrics.json").read_text())
    suite = data["metrics"]["episodic"]
    assert "post_em" in suite
    assert "delta_em" in suite
    assert "post_f1" in suite
    assert "delta_f1" in suite
    assert "post_refusal_rate" in suite
    assert "delta_refusal_rate" in suite
