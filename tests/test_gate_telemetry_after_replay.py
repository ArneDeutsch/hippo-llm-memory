import json
import subprocess
import sys
from pathlib import Path


def test_gate_telemetry_after_replay(tmp_path: Path) -> None:
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
    subprocess.run(base_cmd + ["mode=replay", "persist=true", "replay.cycles=1"], check=True)
    data = json.loads((outdir / "metrics.json").read_text())
    assert data["gates"]["episodic"]["attempts"] > 0
