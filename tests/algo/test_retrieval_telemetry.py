from pathlib import Path

from omegaconf import OmegaConf

from hippo_eval.harness import build_runner, run_suite
from hippo_mem.common.telemetry import registry


def test_retrieval_counters_propagate(tmp_path: Path) -> None:
    registry.reset()
    cfg = OmegaConf.create(
        {
            "suite": "episodic",
            "n": 5,
            "seed": 1337,
            "preset": "configs/eval/memory/hei_nw.yaml",
            "model": "models/tiny-gpt2",
        }
    )
    runner = build_runner(cfg)
    result = run_suite(runner)
    assert result.rows  # retrieval executed
    assert "retrieval" in result.metrics
    epi = result.metrics["retrieval"]["episodic"]
    assert epi["requests"] >= 1
    assert epi["avg_latency_ms"] >= 0.0
