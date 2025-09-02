from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from hippo_mem.eval import harness


def _setup_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, run_id: str) -> DictConfig:
    """Return a minimal config for preflight baseline tests."""

    data_file = tmp_path / "d.jsonl"
    data_file.write_text(json.dumps({"prompt": "p", "answer": "a"}) + "\n")
    monkeypatch.setattr(harness, "_dataset_path", lambda s, n, seed, profile=None: data_file)

    repo_root = Path(__file__).resolve().parents[1]
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


def test_preflight_accepts_both_runid_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid_u = "2025_09_02_50_1337_2025"
    rid_d = rid_u.replace("_", "")
    cfg = _setup_cfg(tmp_path, monkeypatch, run_id=rid_u)
    bdir = Path("runs") / rid_d / "baselines"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "metrics.csv").write_text("suite,em\n")
    outdir = tmp_path / "out"
    try:
        harness.preflight_check(cfg, outdir)
        assert not (outdir / "failed_preflight.json").exists()
    finally:
        shutil.rmtree(Path("runs") / rid_u, ignore_errors=True)
        shutil.rmtree(Path("runs") / rid_d, ignore_errors=True)


def test_preflight_missing_baselines_lists_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid_u = "2025_09_02_50_1337_2025"
    rid_d = rid_u.replace("_", "")
    cfg = _setup_cfg(tmp_path, monkeypatch, run_id=rid_u)
    outdir = tmp_path / "out"
    try:
        with pytest.raises(RuntimeError):
            harness.evaluate(cfg, outdir, preflight=True)
        fail_msg = json.loads((outdir / "failed_preflight.json").read_text())["errors"][0]
        assert f"runs/{rid_u}/baselines/metrics.csv" in fail_msg
        assert f"runs/{rid_d}/baselines/metrics.csv" in fail_msg
    finally:
        shutil.rmtree(Path("runs") / rid_u, ignore_errors=True)
        shutil.rmtree(Path("runs") / rid_d, ignore_errors=True)
