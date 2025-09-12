# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import numpy as np

from hippo_mem.common.telemetry import gate_registry
from hippo_mem.episodic.gating import WriteGate, gate_batch


def test_write_gate_logs_attempts() -> None:
    """Default thresholds trigger both accepts and skips."""

    gate_registry.reset()
    gate = WriteGate()
    probs = np.array([0.9, 0.1])
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    keys = np.array([[1.0, 0.0]], dtype="float32")

    gate_batch(gate, probs, queries, keys)
    stats = gate_registry.get("episodic")
    assert stats.attempts == 2
    assert stats.accepted == 1
    assert stats.skipped == 1
