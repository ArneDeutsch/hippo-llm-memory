# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from types import SimpleNamespace

import numpy as np

from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph


def _sched() -> ReplayScheduler:
    mix = SimpleNamespace(episodic=1.0, semantic=0.0, fresh=0.0)
    store = EpisodicStore(dim=2)
    kg = KnowledgeGraph()
    return ReplayScheduler(store, kg, batch_mix=mix)


def test_salience_over_recency() -> None:
    sched = _sched()
    key = np.zeros(2, dtype="float32")
    sched.add_trace("old_hi", key, score=1.0)
    sched.add_trace("new_lo", key, score=0.0)
    ids = sched.queue.sample(2)
    assert ids[0] == "old_hi"
    assert ids[1] == "new_lo"


def test_recency_breaks_ties() -> None:
    sched = _sched()
    key = np.zeros(2, dtype="float32")
    sched.add_trace("old", key, score=0.5)
    sched.add_trace("new", key, score=0.5)
    ids = sched.queue.sample(2)
    assert ids[0] == "new"


def test_diversity_interacts_with_gradient_overlap() -> None:
    sched = _sched()
    anchor = np.array([1.0, 0.0], dtype="float32")
    sched.add_trace("anchor", anchor, score=0.5)
    sched.add_trace(
        "overlap",
        anchor,
        score=0.5,
        grad_overlap_proxy=-1.0,
    )
    sched.add_trace("diverse", np.array([0.0, 1.0], dtype="float32"), score=0.5)
    ids = sched.queue.sample(3)
    assert ids.index("diverse") < ids.index("overlap")


def test_semantic_batch_ordering(monkeypatch) -> None:
    mix = SimpleNamespace(episodic=0.5, semantic=0.25, fresh=0.25)
    store = EpisodicStore(dim=2)
    kg = KnowledgeGraph()
    sched = ReplayScheduler(store, kg, batch_mix=mix)
    key = np.zeros(2, dtype="float32")
    sched.add_trace("hi", key, score=1.0)
    sched.add_trace("lo", key, score=0.0)
    monkeypatch.setattr("hippo_mem.episodic.replay.random.shuffle", lambda x: None)
    batch = sched.next_batch(4)
    assert batch[0] == ("episodic", "hi")
    assert batch[1] == ("episodic", "lo")
    assert batch[2][0] == "semantic"
    assert batch[3][0] == "fresh"
