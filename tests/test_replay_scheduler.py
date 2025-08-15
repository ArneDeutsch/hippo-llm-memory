"""Tests for the replay scheduler."""

from types import SimpleNamespace

import numpy as np

from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph


def _make_scheduler(mix) -> ReplayScheduler:
    store = EpisodicStore(dim=2)
    kg = KnowledgeGraph()
    return ReplayScheduler(store, kg, batch_mix=mix)


def test_replay_scheduler_batch_mix_counts() -> None:
    """Scheduler returns the configured mix of batch types."""

    mix = SimpleNamespace(episodic=0.5, semantic=0.3, fresh=0.2)
    sched = _make_scheduler(mix)
    sched.add_trace("a", np.array([1.0, 0.0], dtype="float32"), score=0.1)
    sched.add_trace("b", np.array([0.0, 1.0], dtype="float32"), score=0.9)
    batch = sched.next_batch(10)
    kinds = [k for k, _ in batch]
    assert kinds.count("episodic") == 5
    assert kinds.count("semantic") == 3
    assert kinds.count("fresh") == 2
    epi_ids = [tid for k, tid in batch if k == "episodic"]
    assert "b" in epi_ids


def test_replay_scheduler_queue_ranking() -> None:
    """Episodic items are returned according to queue priority."""

    mix = SimpleNamespace(episodic=1.0, semantic=0.0, fresh=0.0)
    sched = _make_scheduler(mix)
    key1 = np.array([1.0, 0.0], dtype="float32")
    key2 = np.array([0.0, 1.0], dtype="float32")
    sched.add_trace("a", key1, score=0.1)
    sched.add_trace("b", key2, score=0.9)
    batch = sched.next_batch(1)
    assert batch[0] == ("episodic", "b")
