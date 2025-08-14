"""Tests for the replay queue."""

import numpy as np

from hippo_mem.episodic.replay import ReplayQueue


def test_replay_queue_weighting() -> None:
    """Queue prioritizes according to configured weights."""

    key1 = np.array([1.0, 0.0], dtype="float32")
    key2 = np.array([0.0, 1.0], dtype="float32")

    # Score-dominated weighting
    q = ReplayQueue(lambda1=1.0, lambda2=0.0, lambda3=0.0)
    q.add("a", key1, score=0.1)
    q.add("b", key2, score=0.9)
    assert q.sample(2)[0] == "b"

    # Recency-dominated weighting
    q = ReplayQueue(lambda1=0.0, lambda2=1.0, lambda3=0.0)
    q.add("a", key1, score=0.0)
    q.add("b", key2, score=0.0)
    assert q.sample(2)[0] == "b"

    # Diversity-dominated weighting
    q = ReplayQueue(lambda1=0.0, lambda2=0.0, lambda3=1.0)
    q.add("a", key1, score=0.0)
    q.add("b", key1, score=0.0)  # identical key -> low diversity
    q.add("c", key2, score=0.0)  # orthogonal -> high diversity
    assert q.sample(3)[0] == "c"
