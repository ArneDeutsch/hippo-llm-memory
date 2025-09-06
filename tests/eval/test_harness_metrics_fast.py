from __future__ import annotations

from omegaconf import OmegaConf

from hippo_eval.harness.metrics import collect_metrics
from hippo_mem.common.gates import GateCounters


def test_collect_metrics_shape() -> None:
    cfg = OmegaConf.create(
        {"suite": "episodic_cross_mem", "preset": "baselines/core", "n": 2, "seed": 1337, "mode": "test"}
    )
    gating = {"episodic": GateCounters(attempts=2, accepted=1)}
    pre_rows = [
        {"memory_hit": 1, "context_match_rate": 1.0},
        {"memory_hit": 0, "context_match_rate": 0.0},
    ]
    metrics = collect_metrics(pre_rows, {"em": 1.0}, [], {"em": 1.0}, cfg, gating=gating)
    suite_metrics = metrics["metrics"]["episodic_cross_mem"]
    assert "pre_em" in suite_metrics and "post_em" in suite_metrics
    assert suite_metrics.get("justification_coverage") == 0.5
    assert suite_metrics.get("gate_accept_rate") == 0.5
