"""Tests for the episodic memory store."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.episodic.gating import WriteGate
from hippo_mem.episodic.replay import ReplayQueue, ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore, TraceValue
from hippo_mem.relational.kg import KnowledgeGraph


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


def test_hopfield_completion_restores_sparse_cue() -> None:
    """Hopfield completion reconstructs the full key from a noisy partial cue."""

    store = EpisodicStore(dim=4)
    full = np.array([1.0, 1.0, 0.0, 0.0], dtype="float32")
    store.write(full, TraceValue(provenance="full"))
    distractor = np.array([0.0, 0.0, 1.0, 1.0], dtype="float32")
    store.write(distractor, TraceValue(provenance="noise"))

    partial = np.array([0.9, 0.1, 0.0, 0.0], dtype="float32")
    completed = store.complete(partial, k=2)

    cos = float(np.dot(completed, full) / (np.linalg.norm(completed) * np.linalg.norm(full)))
    assert cos >= 0.9


def test_gating_threshold_and_pin() -> None:
    """Gating blocks low-salience writes but pin overrides."""

    store = EpisodicStore(dim=4)
    gate = WriteGate(tau=0.5)
    key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    store.write(key, TraceValue(provenance="a"))

    prob = 1.0  # no surprise
    query = key
    decision = gate(prob, query, store.keys())
    if decision.allow:
        store.write(query, TraceValue(provenance="b"))
    assert store.index.ntotal == 1

    decision = gate(prob, query, store.keys(), pin=True)
    if decision.allow:
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


def test_delete_logs_index_error(caplog: pytest.LogCaptureFixture) -> None:
    """Errors removing from FAISS index are logged."""

    store = EpisodicStore(dim=4)
    key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    idx = store.write(key, TraceValue(provenance="alpha"))

    with patch.object(store.index, "remove_ids", side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.ERROR):
            store.delete(idx)
    assert "Failed to remove id" in caplog.text


def test_flash_attention_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter uses scaled_dot_product_attention when enabled."""

    cfg = AdapterConfig(hidden_size=8, num_heads=2, flash_attention=True, enabled=True)
    adapter = EpisodicAdapter(cfg)
    hidden = torch.randn(1, 3, 8)
    traces = torch.randn(1, 4, 8)

    called = {}
    orig = torch.nn.functional.scaled_dot_product_attention

    def fake_sdp_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        called["hit"] = True
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    monkeypatch.setattr(
        "hippo_mem.episodic.adapter.F.scaled_dot_product_attention", fake_sdp_attention
    )

    out = adapter(hidden, traces)
    assert called.get("hit")
    assert out.shape == hidden.shape


def test_update_logs_index_error(caplog: pytest.LogCaptureFixture) -> None:
    """Update logs failures from FAISS removal."""

    store = EpisodicStore(dim=4)
    key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    idx = store.write(key, TraceValue(provenance="alpha"))

    new_key = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
    with patch.object(store.index, "remove_ids", side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.ERROR):
            store.update(idx, key=new_key)
    assert "Failed to remove id" in caplog.text


def test_replay_queue_avoids_consecutive_gradients() -> None:
    """High-overlap gradients are not scheduled back-to-back."""

    queue = ReplayQueue(lambda1=1.0, lambda2=0.0, lambda3=0.0)
    g_a = np.array([1.0, 0.0], dtype="float32")
    g_b = np.array([0.0, 1.0], dtype="float32")
    g_c = np.array([1.0, 0.0], dtype="float32")
    queue.add("a", g_a, score=1.0, grad=g_a)
    queue.add("b", g_b, score=0.8, grad=g_b)
    queue.add("c", g_a, score=0.9, grad=g_c)
    ids = queue.sample(3, grad_sim_threshold=0.99)
    for x, y in zip(ids, ids[1:]):
        assert {x, y} != {"a", "c"}


def test_store_decay_prune_and_rollback() -> None:
    """Decay and prune events can be rolled back."""

    store = EpisodicStore(dim=2)
    k1 = np.array([1.0, 0.0], dtype="float32")
    k2 = np.array([0.0, 1.0], dtype="float32")
    id1 = store.write(k1, TraceValue(provenance="a"))
    id2 = store.write(k2, TraceValue(provenance="b"))

    store.decay(rate=0.5)
    store.prune(min_salience=0.75)
    assert not store.recall(k1, k=1)

    store.rollback(2)
    r1 = store.recall(k1, k=1)[0]
    r2 = store.recall(k2, k=1)[0]
    assert r1.id == id1 and r1.salience == pytest.approx(1.0)
    assert r2.id == id2 and r2.salience == pytest.approx(1.0)

    ops = [e["op"] for e in store._maintenance_log]
    assert ops.count("decay") >= 1
    assert ops.count("prune") >= 1
    assert len(ops) == 4


def test_replay_scheduler_mix_and_unique_ids() -> None:
    """Scheduler respects mix ratios and avoids duplicate episodic ids."""

    mix = SimpleNamespace(episodic=0.5, semantic=0.3, fresh=0.2)
    store = EpisodicStore(dim=2)
    kg = KnowledgeGraph()
    sched = ReplayScheduler(store, kg, batch_mix=mix)

    key = np.array([1.0, 0.0], dtype="float32")
    for i in range(5):
        sched.add_trace(str(i), key, score=1.0 + i)

    batch = sched.next_batch(10)
    kinds = [k for k, _ in batch]
    assert kinds.count("episodic") == 5
    assert kinds.count("semantic") == 3
    assert kinds.count("fresh") == 2

    epi_ids = [tid for k, tid in batch if k == "episodic"]
    assert None not in epi_ids
    assert len(epi_ids) == len(set(epi_ids))


def test_faiss_index_trains_and_queries() -> None:
    """Pending keys trigger index training and subsequent recalls succeed."""

    store = EpisodicStore(dim=4, index_str="IVF1,Flat", train_threshold=5)
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(5, 4)).astype("float32")
    for i, key in enumerate(keys):
        store.write(key, TraceValue(provenance=str(i)))

    assert store.index.is_trained
    res = store.recall(keys[0], k=1)
    assert res and res[0].value.provenance == "0"


@settings(max_examples=25, deadline=None)
@given(
    hnp.arrays(
        np.float32,
        hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=16),
        elements=st.floats(-1.0, 1.0),
    ),
    st.integers(min_value=1, max_value=16),
)
def test_sparse_encode_k_wta_idempotent(vec: np.ndarray, k: int) -> None:
    """Encoding then decoding preserves the top-``k`` indices and values."""

    store = EpisodicStore(dim=vec.size)
    k = min(k, vec.size)
    key = store.sparse_encode(vec, k)
    assert key.indices.size == k
    dense = np.zeros(vec.size, dtype="float32")
    dense[key.indices] = key.values
    key2 = store.sparse_encode(dense, k)
    assert np.array_equal(key.indices, key2.indices)
    assert np.allclose(key.values, key2.values)
