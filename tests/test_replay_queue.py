"""Tests for the replay queue."""

import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hippo_mem.episodic.replay import ReplayQueue


def test_priority_scores_reflect_weights() -> None:
    """Priority scores should reflect weighting configuration."""

    key1 = np.array([1.0, 0.0], dtype="float32")
    key2 = np.array([0.0, 1.0], dtype="float32")
    q = ReplayQueue(lambda1=1.0, lambda2=0.0, lambda3=0.0)
    q.add("a", key1, score=0.1)
    q.add("b", key2, score=0.9)
    priorities = q._priority_scores()
    assert priorities[0] < priorities[1]


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


def test_replay_queue_maxlen_trimming() -> None:
    """Oldest trace is dropped once maxlen is exceeded."""

    key1 = np.array([1.0, 0.0], dtype="float32")
    key2 = np.array([0.0, 1.0], dtype="float32")
    key3 = np.array([1.0, 1.0], dtype="float32")
    q = ReplayQueue(maxlen=2)
    q.add("a", key1, score=0.1)
    q.add("b", key2, score=0.2)
    q.add("c", key3, score=0.3)
    assert [item.trace_id for item in q.items] == ["b", "c"]


@settings(max_examples=25, deadline=None)
@given(
    base=hnp.arrays(np.float32, 2, elements=st.floats(-1.0, 1.0)),
    noise=hnp.arrays(np.float32, 2, elements=st.floats(-0.001, 0.001)),
    other=hnp.arrays(np.float32, 2, elements=st.floats(-1.0, 1.0)),
)
def test_replay_queue_similarity_constraint(
    base: np.ndarray, noise: np.ndarray, other: np.ndarray
) -> None:
    """Highly similar gradients are not scheduled consecutively."""

    if np.linalg.norm(base) == 0:
        base = np.array([1.0, 0.0], dtype="float32")
    v1 = base / np.linalg.norm(base)
    v2 = v1 + noise
    v2 = v2 / np.linalg.norm(v2)
    v3 = other if np.linalg.norm(other) else np.array([0.0, 1.0], dtype="float32")
    v3 = v3 / np.linalg.norm(v3)

    q = ReplayQueue(lambda1=1.0, lambda2=0.0, lambda3=0.0)
    q.add("a", v1, score=1.0, grad=v1)
    q.add("b", v2, score=0.9, grad=v2)
    q.add("c", v3, score=0.8, grad=v3)
    ids = q.sample(3, grad_sim_threshold=0.95)
    assert "a" in ids
    assert "b" not in ids
