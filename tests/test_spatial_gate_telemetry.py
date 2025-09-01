"""Telemetry integration test for spatial gating."""

from hippo_mem.common.telemetry import gate_registry
from hippo_mem.eval.harness import EvalConfig, run_suite


def test_spatial_gate_telemetry() -> None:
    gate_registry.reset()
    cfg = EvalConfig(
        suite="spatial",
        n=5,
        seed=1337,
        preset="configs/eval/memory/smpd.yaml",
        model="models/tiny-gpt2",
        replay_cycles=1,
    )
    _rows, metrics, _ = run_suite(cfg)
    gating = metrics["gating"]["spatial"]
    assert gating["attempts"] > 0
    assert gating["accepted"] >= 0
