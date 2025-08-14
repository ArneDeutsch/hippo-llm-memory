"""Tests for the episodic memory store."""

import numpy as np
import pytest

from hippo_mem.episodic.gating import WriteGate
from hippo_mem.episodic.store import EpisodicStore, TraceValue


def test_one_shot_write_recall() -> None:
    """A written item can be recalled exactly."""

    store = EpisodicStore(dim=4)
    key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    store.write(key, TraceValue(provenance="alpha"))

    results = store.recall(key, k=1)
    assert results and results[0].value.provenance == "alpha"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_partial_cue_recall_under_distractors() -> None:
    """Partial cues retrieve the target despite random distractors."""

    dim = 4
    store = EpisodicStore(dim=dim)
    target_key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    store.write(target_key, TraceValue(provenance="target"))

    rng = np.random.default_rng(0)
    for i in range(5):
        distractor = rng.normal(size=dim).astype("float32")
        store.write(distractor, TraceValue(provenance=f"d{i}"))

    query = np.array([0.9, 0.1, 0.0, 0.0], dtype="float32")
    results = store.recall(query, k=1)
    assert results and results[0].value.provenance == "target"


def test_gating_threshold_and_pin() -> None:
    """Gating blocks low-salience writes but pin overrides."""

    store = EpisodicStore(dim=4)
    gate = WriteGate(tau=0.5)
    key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    store.write(key, TraceValue(provenance="a"))

    prob = 1.0  # no surprise
    query = key
    allow, _ = gate(prob, query, store.keys())
    if allow:
        store.write(query, TraceValue(provenance="b"))
    assert store.index.ntotal == 1

    allow, _ = gate(prob, query, store.keys(), pin=True)
    if allow:
        store.write(query, TraceValue(provenance="b"))
    assert store.index.ntotal == 2


def test_delete_removes_trace() -> None:
    """Deleting a trace removes it from recall."""

    store = EpisodicStore(dim=4)
    key1 = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    key2 = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
    id1 = store.write(key1, TraceValue(provenance="alpha"))
    id2 = store.write(key2, TraceValue(provenance="beta"))

    store.delete(id1)
    results = store.recall(key1, k=2)
    assert all(r.id != id1 for r in results)

    results2 = store.recall(key2, k=1)
    assert results2 and results2[0].id == id2
