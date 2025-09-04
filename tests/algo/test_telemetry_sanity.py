import pytest
from omegaconf import OmegaConf

from hippo_eval.eval import harness
from hippo_mem.common.telemetry import (
    record_stats,
    registry,
    set_strict_telemetry,
    validate_retrieval_snapshot,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    registry.reset()
    for snap in registry.all_snapshots().values():
        assert snap["requests"] == 0
        assert snap["hits_at_k"] == 0
        assert snap["total_k"] == 0
        assert snap["tokens_returned"] == 0
        assert snap["avg_latency_ms"] == 0.0
        assert snap["hit_rate_at_k"] == 0.0
        assert snap["k"] == 0
        assert snap["batch_size"] == 0


def test_valid_stats_pass() -> None:
    set_strict_telemetry(True)
    record_stats("episodic", k=4, batch_size=1, hits=2, tokens=4, latency_ms=0.1)
    snap = registry.get("episodic").snapshot()
    assert snap["requests"] == 1 and snap["hits_at_k"] == 2
    validate_retrieval_snapshot(snap)
    set_strict_telemetry(False)


def test_invalid_hits_raise() -> None:
    set_strict_telemetry(True)
    with pytest.raises(ValueError):
        record_stats("episodic", k=4, batch_size=1, hits=5, tokens=4, latency_ms=0.1)
    set_strict_telemetry(False)


def test_hit_rate_mismatch_raises() -> None:
    set_strict_telemetry(True)
    with pytest.raises(ValueError):
        validate_retrieval_snapshot({"k": 1, "requests": 10, "hits_at_k": 5, "hit_rate_at_k": 0.9})
    set_strict_telemetry(False)


def test_env_var_enables_strict(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STRICT_TELEMETRY", "1")
    cfg = OmegaConf.create(
        {
            "suite": "missing",
            "n": 1,
            "seed": 0,
            "preset": "baselines/core",
            "model": "models/tiny-gpt2",
            "dataset_profile": None,
        }
    )
    with pytest.raises(FileNotFoundError):
        harness.evaluate(cfg, tmp_path)
    with pytest.raises(ValueError):
        validate_retrieval_snapshot({"k": 1, "requests": 1, "hits_at_k": 2, "hit_rate_at_k": 0.0})
    set_strict_telemetry(False)
