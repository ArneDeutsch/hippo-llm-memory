# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from hippo_eval.eval import harness
from hippo_mem.common.telemetry import gate_registry


def _setup_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, run_id: str) -> DictConfig:
    """Return a minimal config for preflight baseline tests."""

    data_file = tmp_path / "d.jsonl"
    data_file.write_text(json.dumps({"prompt": "p", "answer": "a"}) + "\n")
    monkeypatch.setattr(
        harness, "_dataset_path", lambda s, n, seed, profile=None, mode=None: data_file
    )

    repo_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(repo_root / "configs" / "eval" / "default.yaml")
    cfg.suite = "episodic"
    cfg.preset = "memory/hei_nw"
    cfg.model = str(repo_root / "models" / "tiny-gpt2")
    cfg.n = 0
    cfg.seed = 0
    cfg.mode = "test"
    cfg.store_dir = str(tmp_path / "stores" / "hei_nw")
    cfg.session_id = "sid"
    cfg.run_id = run_id
    cfg = harness._load_preset(cfg)
    cfg = harness._apply_model_defaults(cfg)

    store_path = Path(cfg.store_dir) / cfg.session_id
    store_path.mkdir(parents=True, exist_ok=True)
    (store_path / "store_meta.json").write_text(
        json.dumps({"schema": "episodic.store_meta.v1", "replay_samples": 1, "source": "replay"})
    )
    (store_path / "episodic.jsonl").write_text("")

    return cfg


def test_preflight_missing_baselines_lists_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = "2025_09_02_50_1337_2025"
    cfg = _setup_cfg(tmp_path, monkeypatch, run_id=rid)
    outdir = tmp_path / "out"
    try:
        with pytest.raises(RuntimeError):
            harness.evaluate(cfg, outdir, preflight=True)
        fail_msg = json.loads((outdir / "failed_preflight.json").read_text())["errors"][0]
        expect = f"runs/{rid}/baselines/metrics.csv"
        assert expect in fail_msg
        assert f"python -m hippo_eval.baselines --run-id {rid}" in fail_msg
        attempts = sum(
            gate_registry.get(name).attempts for name in ("episodic", "relational", "spatial")
        )
        assert attempts > 0
    finally:
        shutil.rmtree(Path("runs") / rid, ignore_errors=True)


def test_preflight_missing_store_meta_lists_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = "2025_09_02_50_1337_2025"
    cfg = _setup_cfg(tmp_path, monkeypatch, run_id=rid)
    bdir = Path("runs") / rid / "baselines"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "metrics.csv").write_text("suite,em\n")
    store_dir = Path(cfg.store_dir)
    meta_algo = store_dir / cfg.session_id / "store_meta.json"
    meta_algo.unlink()
    outdir = tmp_path / "out"
    try:
        with pytest.raises(RuntimeError):
            harness.evaluate(cfg, outdir, preflight=True)
        fail_msg = json.loads((outdir / "failed_preflight.json").read_text())["errors"][0]
        meta_base = store_dir / "hei_nw" / cfg.session_id / "store_meta.json"
        assert str(meta_algo) in fail_msg
        assert str(meta_base) in fail_msg
        attempts = sum(
            gate_registry.get(name).attempts for name in ("episodic", "relational", "spatial")
        )
        assert attempts > 0
    finally:
        shutil.rmtree(Path("runs") / rid, ignore_errors=True)
