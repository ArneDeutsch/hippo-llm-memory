"""Baseline evaluation robustness tests."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from hippo_eval.eval.harness import _apply_model_defaults, evaluate


def test_baseline_metrics_in_teach(tmp_path: Path) -> None:
    cfg = OmegaConf.load("configs/eval/default.yaml")
    cfg.suite = "episodic"
    cfg.preset = "baselines/core"
    cfg.model = "models/tiny-gpt2"
    cfg.n = 1
    cfg.seed = 1337
    cfg.dry_run = True
    cfg.mode = "teach"
    cfg = _apply_model_defaults(cfg)

    outdir = tmp_path / "run"
    evaluate(cfg, outdir)

    metrics = json.loads((outdir / "metrics.json").read_text())
    suite_metrics = metrics["metrics"][cfg.suite]
    for key in ("pre_em", "pre_em_raw", "pre_em_norm", "pre_f1"):
        assert key in suite_metrics and suite_metrics[key] is not None
