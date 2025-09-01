import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _write_store(base: Path) -> Path:
    session = base / "hei_nw" / "sid"
    session.mkdir(parents=True, exist_ok=True)
    rec = {"prompt": "p", "answer": "a", "salience": 1.0, "usage": 0, "ts": 0.0}
    (session / "episodic.jsonl").write_text(json.dumps(rec) + "\n")
    (session / "relational.jsonl").write_text("")
    (session / "spatial.jsonl").write_text("")
    meta = {"replay_samples": 1, "source": "replay"}
    (session / "store_meta.json").write_text(json.dumps(meta))
    return base


@pytest.mark.slow
def test_test_consolidation_pre_post(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_MODEL_PATH": "models/tiny-gpt2",
        }
    )

    # build adapter
    store_dir = _write_store(tmp_path / "stores")
    adapter_dir = tmp_path / "adapter"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "peft:\n  rank: 2\n  alpha: 2\n  dropout: 0.0\n  targets: [c_attn]\n"
        "train:\n  lr: 1.0e-4\n  steps: 1\n  batch_size: 1\n"
    )
    cmd = [
        sys.executable,
        str(repo / "scripts" / "replay_consolidate.py"),
        "--store_dir",
        str(store_dir),
        "--session_id",
        "sid",
        "--config",
        str(cfg),
        "--outdir",
        str(adapter_dir),
        "--model",
        "models/tiny-gpt2",
    ]
    subprocess.run(cmd, check=True, env=env, cwd=repo)

    pre_dir = tmp_path / "pre"
    cmd_pre = [
        sys.executable,
        str(repo / "scripts" / "test_consolidation.py"),
        "--phase",
        "pre",
        "--suite",
        "episodic",
        "--n",
        "1",
        "--seed",
        "1337",
        "--model",
        "models/tiny-gpt2",
        "--allow-tiny-test-model",
        "--outdir",
        str(pre_dir),
    ]
    subprocess.run(cmd_pre, check=True, env=env, cwd=repo)

    post_dir = tmp_path / "post"
    cmd_post = [
        sys.executable,
        str(repo / "scripts" / "test_consolidation.py"),
        "--phase",
        "post",
        "--suite",
        "episodic",
        "--n",
        "1",
        "--seed",
        "1337",
        "--model",
        "models/tiny-gpt2",
        "--allow-tiny-test-model",
        "--adapter",
        str(adapter_dir),
        "--pre_dir",
        str(pre_dir),
        "--outdir",
        str(post_dir),
        "--min-em-uplift",
        "0",
    ]
    subprocess.run(cmd_post, check=True, env=env, cwd=repo)

    metrics = json.loads((post_dir / "metrics.json").read_text())
    assert "delta" in metrics
    assert isinstance(metrics["delta"].get("em"), float)
