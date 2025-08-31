import pytest

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
        assert snap["hits"] == 0
        assert snap["total_k"] == 0
        assert snap["tokens_returned"] == 0
        assert snap["avg_latency_ms"] == 0.0
        assert snap["hit_rate_at_k"] == 0.0


def test_valid_stats_pass() -> None:
    set_strict_telemetry(True)
    record_stats("episodic", k=4, hits=2, tokens=4, latency_ms=0.1)
    # Should not raise
    validate_retrieval_snapshot(registry.get("episodic").snapshot())
    set_strict_telemetry(False)


def test_invalid_hits_raise() -> None:
    set_strict_telemetry(True)
    with pytest.raises(ValueError):
        record_stats("episodic", k=4, hits=5, tokens=4, latency_ms=0.1)
    set_strict_telemetry(False)


def test_hit_rate_mismatch_raises() -> None:
    set_strict_telemetry(True)
    with pytest.raises(ValueError):
        validate_retrieval_snapshot({"total_k": 10, "hits": 5, "hit_rate_at_k": 0.9})
    set_strict_telemetry(False)
