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
