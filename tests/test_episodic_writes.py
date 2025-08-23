import numpy as np
import pytest

from hippo_mem.episodic.async_writer import AsyncStoreWriter
from hippo_mem.episodic.gating import WriteGate, gate_batch
from hippo_mem.episodic.store import TraceValue


class DummyStore:
    def __init__(self):
        self.written = []

    def write(self, key, value):
        self.written.append((key, value))
        return len(self.written)


def test_gate_batch_deterministic():
    gate = WriteGate(tau=0.5)
    probs = np.array([0.1, 0.9], dtype=float)
    queries = np.ones((2, 1), dtype=np.float32)
    keys = np.ones((1, 1), dtype=np.float32)
    decisions, rate = gate_batch(gate, probs, queries, keys)
    assert [d.allow for d in decisions] == [True, False]
    assert rate == 0.5


def test_gate_batch_applies_reward_and_pin() -> None:
    """Reward and pin flags influence write decisions."""

    gate = WriteGate(tau=0.5, alpha=0.0, beta=0.0, gamma=1.0, delta=1.0)
    probs = np.ones(3, dtype=float)
    queries = np.zeros((3, 1), dtype=np.float32)
    keys = np.zeros((0, 1), dtype=np.float32)
    rewards = np.array([0.0, 1.0, 0.0], dtype=float)
    pins = np.array([False, False, True])
    decisions, rate = gate_batch(gate, probs, queries, keys, rewards, pins)
    assert [d.allow for d in decisions] == [False, True, True]
    assert rate == pytest.approx(2 / 3)


def test_async_writer_enqueues():
    store = DummyStore()
    writer = AsyncStoreWriter(store, maxsize=2)
    gate = WriteGate(tau=0.5)
    key = np.zeros(1, dtype=np.float32)
    dec = gate(0.1, key, np.zeros((0, 1), dtype=np.float32))
    if dec.allow:
        writer.enqueue(key, TraceValue(provenance="unit"))
    writer.queue.join()
    writer.stop()
    assert writer.stats["writes_enqueued"] == 1
    assert writer.stats["writes_committed"] == 1
    assert len(store.written) == 1
