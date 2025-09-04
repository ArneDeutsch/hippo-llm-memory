import json
import subprocess
import sys
from pathlib import Path


def test_gate_telemetry_after_replay(tmp_path: Path) -> None:
    store_dir = tmp_path / "stores"
    outdir = tmp_path / "run"
    base_cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=1",
        "seed=1337",
        "model=models/tiny-gpt2",
        f"store_dir={store_dir}",
        "session_id=s1",
    ]
    subprocess.run(
        base_cmd + [f"outdir={outdir}", "mode=teach", "persist=true", "dry_run=true"],
        check=True,
    )
    teach_data = json.loads((outdir / "metrics.json").read_text())
    subprocess.run(
        base_cmd
        + [
            f"outdir={outdir}",
            "mode=replay",
            "persist=true",
            "replay.cycles=1",
            "dry_run=true",
        ],
        check=True,
    )
    replay_data = json.loads((outdir / "metrics.json").read_text())
    before = teach_data["gating"]["episodic"]["attempts"]
    after = replay_data["gating"]["episodic"]["attempts"]
    assert before > 0
    assert after == before
