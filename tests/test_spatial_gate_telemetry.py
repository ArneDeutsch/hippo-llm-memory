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
    gates = metrics["gates"]["spatial"]
    assert gates["attempts"] > 0
    assert gates["accepts"] >= 0
