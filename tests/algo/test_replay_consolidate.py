# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hippo_mem.consolidation.trainer import parse_args


def _write_store(base: Path) -> Path:
    session = base / "hei_nw" / "s"
    session.mkdir(parents=True, exist_ok=True)
    epi = session / "episodic.jsonl"
    rec = {"prompt": "p", "answer": "a", "salience": 1.0, "usage": 0, "ts": 0.0}
    epi.write_text(json.dumps(rec) + "\n")
    (session / "relational.jsonl").write_text("")
    (session / "spatial.jsonl").write_text("")
    return base / "hei_nw"


def test_parse_args(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("train:\n  steps: 1\n")
    args = parse_args(
        [
            "--store_dir",
            "sdir",
            "--session_id",
            "s",
            "--outdir",
            "o",
            "--config",
            str(cfg),
        ]
    )
    assert args.store_dir == "sdir"
    assert args.session_id == "s"
    assert args.outdir == "o"
    assert args.config == str(cfg)


@pytest.mark.slow
def test_replay_consolidate_auto_targets(tmp_path: Path) -> None:
    store_dir = _write_store(tmp_path / "stores")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "peft:\n  rank: 2\n  alpha: 2\n  dropout: 0.0\n  targets: auto\n"
        "train:\n  lr: 1.0e-4\n  steps: 1\n  batch_size: 1\n"
    )
    outdir = tmp_path / "out"
    env = os.environ.copy()
    env.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_MODEL_PATH": "models/tiny-gpt2",
        }
    )
    cmd = [
        sys.executable,
        "-m",
        "hippo_mem.consolidation.trainer",
        "--store_dir",
        str(store_dir),
        "--session_id",
        "s",
        "--config",
        str(cfg),
        "--outdir",
        str(outdir),
        "--model",
        "models/tiny-gpt2",
    ]
    subprocess.run(cmd, check=True, env=env)
    cfg_path = outdir / "adapter_config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert "target_modules" in cfg and len(cfg["target_modules"]) > 0


@pytest.mark.slow
def test_replay_consolidate_runs(tmp_path: Path) -> None:
    store_dir = _write_store(tmp_path / "stores")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "peft:\n  rank: 2\n  alpha: 2\n  dropout: 0.0\n  targets: [c_attn]\n"
        "train:\n  lr: 1.0e-4\n  steps: 1\n  batch_size: 1\n"
    )
    outdir = tmp_path / "out"
    env = os.environ.copy()
    env.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_MODEL_PATH": "models/tiny-gpt2",
        }
    )
    cmd = [
        sys.executable,
        "-m",
        "hippo_mem.consolidation.trainer",
        "--store_dir",
        str(store_dir),
        "--session_id",
        "s",
        "--config",
        str(cfg),
        "--outdir",
        str(outdir),
        "--model",
        "models/tiny-gpt2",
    ]
    subprocess.run(cmd, check=True, env=env)
    meta = json.loads((outdir / "meta.json").read_text())
    assert meta["replay_samples"] >= 1
    assert len(meta["lora_config_hash"]) > 10


@pytest.mark.slow
def test_replay_consolidate_merge(tmp_path: Path) -> None:
    store_dir = _write_store(tmp_path / "stores")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "peft:\n  rank: 2\n  alpha: 2\n  dropout: 0.0\n  targets: [c_attn]\n"
        "train:\n  lr: 1.0e-4\n  steps: 1\n  batch_size: 1\n"
    )
    outdir = tmp_path / "out"
    env = os.environ.copy()
    env.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_MODEL_PATH": "models/tiny-gpt2",
        }
    )
    cmd = [
        sys.executable,
        "-m",
        "hippo_mem.consolidation.trainer",
        "--store_dir",
        str(store_dir),
        "--session_id",
        "s",
        "--config",
        str(cfg),
        "--outdir",
        str(outdir),
        "--model",
        "models/tiny-gpt2",
        "--merge",
    ]
    subprocess.run(cmd, check=True, env=env)
    has_bin = (outdir / "pytorch_model.bin").exists() or (outdir / "model.safetensors").exists()
    assert has_bin
    assert not (outdir / "adapter_model.bin").exists()
