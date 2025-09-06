"""Telemetry integration test for spatial gating."""

from omegaconf import OmegaConf

from hippo_eval.harness import build_runner
from hippo_mem.common.telemetry import gate_registry


def test_spatial_gate_telemetry() -> None:
    gate_registry.reset()
    cfg = OmegaConf.create(
        {
            "suite": "spatial_multi",
            "n": 5,
            "seed": 1337,
            "preset": "configs/eval/memory/smpd.yaml",
            "model": "models/tiny-gpt2",
            "replay_cycles": 1,
        }
    )
    runner = build_runner(cfg)
    result = runner.run()
    gating = result.metrics["gating"]["spatial"]
    assert gating["attempts"] > 0
    assert gating["accepted"] >= 0
