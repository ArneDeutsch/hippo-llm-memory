from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from hippo_mem.eval import harness


def _setup_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, baseline: bool) -> DictConfig:
    """Return a minimal config for preflight tests."""

    data_file = tmp_path / "d.jsonl"
    data_file.write_text(json.dumps({"prompt": "p", "answer": "a"}) + "\n")
    monkeypatch.setattr(harness, "_dataset_path", lambda s, n, seed, profile=None: data_file)

    cfg = OmegaConf.load("configs/eval/default.yaml")
    cfg.suite = "episodic"
    cfg.preset = "memory/hei_nw"
    cfg.model = "models/tiny-gpt2"
    cfg.n = 0
    cfg.seed = 0
    cfg.mode = "test"
    cfg.store_dir = str(tmp_path / "stores" / "hei_nw")
    cfg.session_id = "sid"
    cfg = harness._load_preset(cfg)
    cfg = harness._apply_model_defaults(cfg)

    store_path = Path(cfg.store_dir) / cfg.session_id
    store_path.mkdir(parents=True, exist_ok=True)
    (store_path / "store_meta.json").write_text(
        json.dumps(
            {
                "schema": "episodic.store_meta.v1",
                "replay_samples": 1,
                "source": "replay",
            }
        )
    )
    (store_path / "episodic.jsonl").write_text("")

    if baseline:
        bdir = tmp_path.parent.parent / "baselines"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "metrics.csv").write_text("suite,em\n")
    else:
        bdir = tmp_path.parent.parent / "baselines"
        if bdir.exists():
            shutil.rmtree(bdir)

    return cfg


def test_preflight_fails_without_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _setup_cfg(tmp_path, monkeypatch, baseline=False)
    outdir = tmp_path / "mem" / "episodic"
    with pytest.raises(RuntimeError):
        harness.evaluate(cfg, outdir, preflight=True)
    assert (outdir / "failed_preflight.json").exists()


def test_preflight_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _setup_cfg(tmp_path, monkeypatch, baseline=True)
    outdir = tmp_path / "mem" / "episodic"
    harness.preflight_check(cfg, outdir)
    assert not (outdir / "failed_preflight.json").exists()
