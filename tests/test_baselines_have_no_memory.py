"""Ensure baseline presets do not instantiate memory modules."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from hippo_mem.eval.bench import _init_modules
from hippo_mem.eval.harness import _apply_model_defaults, evaluate


def test_baseline_has_no_memory(tmp_path: Path) -> None:
    """`baselines/core` should run without any memory modules or telemetry."""

    cfg = OmegaConf.load("configs/eval/default.yaml")
    cfg.suite = "episodic"
    cfg.preset = "baselines/core"
    cfg.model = "models/tiny-gpt2"
    cfg.n = 1
    cfg.seed = 1337
    cfg.dry_run = True
    cfg = _apply_model_defaults(cfg)

    modules = _init_modules(cfg.get("memory"), {})
    assert modules == {}

    outdir = tmp_path / "run"
    evaluate(cfg, outdir)

    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics["store"]["size"] == 0
    assert all(m["requests"] == 0 for m in metrics["retrieval"].values())
