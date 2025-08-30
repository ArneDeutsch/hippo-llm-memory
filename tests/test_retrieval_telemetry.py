from pathlib import Path

from hippo_mem.common.telemetry import registry
from hippo_mem.eval.harness import EvalConfig, run_suite


def test_retrieval_counters_propagate(tmp_path: Path) -> None:
    registry.reset()
    cfg = EvalConfig(
        suite="episodic",
        n=5,
        seed=1337,
        preset="configs/eval/memory/hei_nw.yaml",
        model="models/tiny-gpt2",
    )
    rows, metrics, _ = run_suite(cfg)
    assert rows  # retrieval executed
    assert "retrieval" in metrics
    epi = metrics["retrieval"]["episodic"]
    assert epi["requests"] >= 1
    assert epi["avg_latency_ms"] >= 0.0
