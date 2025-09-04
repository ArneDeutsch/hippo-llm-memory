from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from hippo_eval.eval import harness
from hippo_mem.common.telemetry import gate_registry


def _setup_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, baseline: bool, store_has_data: bool = True
) -> DictConfig:
    """Return a minimal config for preflight tests."""

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
    cfg.run_id = "testrun"
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
    if store_has_data:
        (store_path / "episodic.jsonl").write_text("{}\n")
    else:
        (store_path / "episodic.jsonl").write_text("")

    bdir = Path("runs") / cfg.run_id / "baselines"
    if baseline:
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "metrics.csv").write_text("suite,em\n")
    else:
        if bdir.parent.exists():
            shutil.rmtree(bdir.parent)

    return cfg


def test_preflight_fails_without_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _setup_cfg(tmp_path, monkeypatch, baseline=False)
    outdir = tmp_path / "mem" / "episodic"
    try:
        with pytest.raises(RuntimeError):
            harness.evaluate(cfg, outdir, preflight=True)
        assert (outdir / "failed_preflight.json").exists()
    finally:
        shutil.rmtree(Path("runs") / cfg.run_id, ignore_errors=True)


def test_preflight_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _setup_cfg(tmp_path, monkeypatch, baseline=True)
    outdir = tmp_path / "mem" / "episodic"
    try:
        harness.preflight_check(cfg, outdir)
        assert not (outdir / "failed_preflight.json").exists()
        attempts = sum(
            gate_registry.get(name).attempts for name in ("episodic", "relational", "spatial")
        )
        assert attempts > 0
    finally:
        shutil.rmtree(Path("runs") / cfg.run_id, ignore_errors=True)


def test_preflight_missing_baseline_hints_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _setup_cfg(tmp_path, monkeypatch, baseline=False)
    outdir = tmp_path / "mem" / "episodic"
    try:
        with pytest.raises(RuntimeError):
            harness.evaluate(cfg, outdir, preflight=True)
        fail_msg = json.loads((outdir / "failed_preflight.json").read_text())["errors"][0]
        assert f"runs/{cfg.run_id}/baselines/metrics.csv" in fail_msg
        assert f"python scripts/run_baselines.py --run-id {cfg.run_id}" in fail_msg
        attempts = sum(
            gate_registry.get(name).attempts for name in ("episodic", "relational", "spatial")
        )
        assert attempts > 0
    finally:
        shutil.rmtree(Path("runs") / cfg.run_id, ignore_errors=True)


def test_preflight_empty_store_hints_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _setup_cfg(tmp_path, monkeypatch, baseline=True, store_has_data=False)
    outdir = tmp_path / "mem" / "episodic"
    try:
        with pytest.raises(RuntimeError):
            harness.evaluate(cfg, outdir, preflight=True)
        fail_msg = json.loads((outdir / "failed_preflight.json").read_text())["errors"][0]
        assert "empty store" in fail_msg
        assert "python scripts/eval_model.py --mode teach" in fail_msg
        assert f"--run-id {cfg.run_id}" in fail_msg
        attempts = sum(
            gate_registry.get(name).attempts for name in ("episodic", "relational", "spatial")
        )
        assert attempts > 0
    finally:
        shutil.rmtree(Path("runs") / cfg.run_id, ignore_errors=True)


def test_preflight_gate_attempts_zero_hints_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _setup_cfg(tmp_path, monkeypatch, baseline=True, store_has_data=True)
    outdir = tmp_path / "mem" / "episodic"

    def _stub_eval(cfg, out, preflight=False):
        return None

    monkeypatch.setattr(harness, "evaluate", _stub_eval)

    try:
        with pytest.raises(RuntimeError):
            harness.preflight_check(cfg, outdir)
        fail_msg = json.loads((outdir / "failed_preflight.json").read_text())["errors"][0]
        assert "gate.attempts == 0" in fail_msg
        assert "python scripts/eval_model.py --mode teach" in fail_msg
        assert f"--run-id {cfg.run_id}" in fail_msg
    finally:
        shutil.rmtree(Path("runs") / cfg.run_id, ignore_errors=True)
