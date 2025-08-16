"""Tests for the episodic memory store."""

import logging
from unittest.mock import patch

import numpy as np
import pytest
import torch

from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.episodic.gating import WriteGate
from hippo_mem.episodic.replay import ReplayQueue
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
