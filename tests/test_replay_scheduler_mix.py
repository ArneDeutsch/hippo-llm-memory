from types import SimpleNamespace

import numpy as np

from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph


def test_batch_mix_stable_over_cycles() -> None:
    """Repeated batches match configured mix ratios."""

    mix = SimpleNamespace(episodic=0.5, semantic=0.3, fresh=0.2)
    store = EpisodicStore(dim=2)
    kg = KnowledgeGraph()
    sched = ReplayScheduler(store, kg, batch_mix=mix)
    key = np.array([1.0, 0.0], dtype="float32")
    for i in range(10):
        sched.add_trace(f"t{i}", key, score=1.0)
    kg.upsert(
        "H",
        "rel",
        "T",
        "ctx",
        head_embedding=[1.0, 0.0],
        tail_embedding=[0.0, 1.0],
    )
    counts = {"episodic": 0, "semantic": 0, "fresh": 0}
    for _ in range(100):
        for kind, _ in sched.next_batch(10):
            counts[kind] += 1
    total = sum(counts.values())
    assert abs(counts["episodic"] / total - 0.5) < 0.05
    assert abs(counts["semantic"] / total - 0.3) < 0.05
    assert abs(counts["fresh"] / total - 0.2) < 0.05
