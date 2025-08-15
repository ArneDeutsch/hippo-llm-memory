"""Tests for the priority replay scheduler."""

import numpy as np

from hippo_mem.episodic.replay import ReplayQueue
from scripts.train_lora import TrainConfig
from hippo_mem.consolidation.scheduler import PriorityReplayScheduler


def _make_scheduler():
    q = ReplayQueue()
    key = np.zeros(2, dtype=np.float32)
    q.add("t1", key, score=0.1)
    q.add("t2", key, score=0.2)
    mix = TrainConfig.BatchMix(episodic=0.5, semantic=0.3, fresh=0.2)
    scheduler = PriorityReplayScheduler(q, mix)
    return scheduler


def test_schedule_honours_mix_and_samples_ids():
    """Scheduler respects the requested mix and samples from the queue."""

    scheduler = _make_scheduler()
    schedule = scheduler.schedule(total=10, batch_size=1)
    counts = {"episodic": 0, "semantic": 0, "fresh": 0}
    for kind, ids in schedule:
        counts[kind] += 1
        if kind == "episodic":
            assert ids[0] in {"t1", "t2"}
    assert counts["episodic"] == 5
    assert counts["semantic"] == 3
    assert counts["fresh"] == 2
