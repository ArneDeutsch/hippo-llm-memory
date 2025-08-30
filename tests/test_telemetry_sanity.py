import pytest

from hippo_mem.common.telemetry import (
    record_stats,
    registry,
    set_strict_telemetry,
    validate_retrieval_snapshot,
)


def test_valid_stats_pass() -> None:
    registry.reset()
    set_strict_telemetry(True)
    record_stats("episodic", k=4, hits=2, tokens=4, latency_ms=0.1)
    # Should not raise
    validate_retrieval_snapshot(registry.get("episodic").snapshot())
    set_strict_telemetry(False)


def test_invalid_hits_raise() -> None:
    registry.reset()
    set_strict_telemetry(True)
    with pytest.raises(ValueError):
        record_stats("episodic", k=4, hits=5, tokens=4, latency_ms=0.1)
    set_strict_telemetry(False)


def test_hit_rate_mismatch_raises() -> None:
    set_strict_telemetry(True)
    with pytest.raises(ValueError):
        validate_retrieval_snapshot({"total_k": 10, "hits": 5, "hit_rate_at_k": 0.9})
    set_strict_telemetry(False)
