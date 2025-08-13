"""Tests for episodic memory stubs."""

from hippo_mem.episodic.gating import gate_event
from hippo_mem.episodic.replay import replay
from hippo_mem.episodic.store import EpisodicStore


def test_episodic_flow() -> None:
    """Events gated and replayed correctly."""
    store = EpisodicStore()
    if gate_event("event"):
        store.add("event")
    assert replay(store.events) == ["event"]
