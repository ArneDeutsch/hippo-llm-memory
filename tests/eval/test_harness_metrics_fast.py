from __future__ import annotations

from omegaconf import OmegaConf

from hippo_eval.harness.metrics import collect_metrics
from hippo_mem.common.gates import GateCounters


def test_collect_metrics_shape() -> None:
    cfg = OmegaConf.create(
        {"suite": "episodic", "preset": "baselines/core", "n": 2, "seed": 1337, "mode": "test"}
    )
    gating = {"episodic": GateCounters(attempts=1, accepted=1)}
    metrics = collect_metrics([], {"em": 1.0}, [], {"em": 1.0}, cfg, gating=gating)
    suite_metrics = metrics["metrics"]["episodic"]
    assert "pre_em" in suite_metrics and "post_em" in suite_metrics
