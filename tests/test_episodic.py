"""Tests for the episodic memory store."""

import numpy as np
import pytest

from hippo_mem.episodic.store import EpisodicStore


def test_one_shot_write_recall() -> None:
    """A written item can be recalled exactly."""

    store = EpisodicStore(dim=4)
    key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    store.write(key, "alpha")

    results = store.recall(key, k=1)
    assert results and results[0].value == "alpha"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_partial_cue_recall_under_distractors() -> None:
    """Partial cues retrieve the target despite random distractors."""

    dim = 4
    store = EpisodicStore(dim=dim)
    target_key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    store.write(target_key, "target")

    rng = np.random.default_rng(0)
    for i in range(5):
        distractor = rng.normal(size=dim).astype("float32")
        store.write(distractor, f"d{i}")

    query = np.array([0.9, 0.1, 0.0, 0.0], dtype="float32")
    results = store.recall(query, k=1)
    assert results and results[0].value == "target"
