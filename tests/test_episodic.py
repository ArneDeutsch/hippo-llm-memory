"""Tests for the episodic memory store."""

import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.episodic.gating import WriteGate, k_wta, surprise
from hippo_mem.episodic.replay import ReplayQueue, ReplayScheduler
from hippo_mem.episodic.retrieval import episodic_retrieve_and_pack
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.episodic.types import TraceValue
from hippo_mem.episodic.utils import cosine_dissimilarity
from hippo_mem.relational.kg import KnowledgeGraph


def test_cosine_dissimilarity_reductions() -> None:
    """Utility computes max and mean cosine dissimilarity."""

    vec = np.array([1.0, 0.0], dtype="float32")
    mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    max_val = cosine_dissimilarity(vec, mat, "max")
    mean_val = cosine_dissimilarity(vec, mat, "mean")
    assert max_val == pytest.approx(0.0)
    assert mean_val == pytest.approx(0.5)


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


def test_noisy_cue_completed_and_recalled() -> None:
    """A noisy cue is completed and improves recall cosine similarity."""

    store = EpisodicStore(dim=4, config={"hopfield": True})
    full = np.array([1.0, 1.0, 0.0, 0.0], dtype="float32")
    store.write(full, TraceValue(provenance="full"))
    noisy = np.array([0.8, 0.2, 0.0, 0.0], dtype="float32")

    base = store.recall(noisy, k=1)[0]
    hidden = torch.from_numpy(noisy).view(1, 1, -1)
    spec = SimpleNamespace(k=1)
    episodic_retrieve_and_pack(hidden, spec, store, torch.nn.Identity())
    completed = store.complete(noisy, k=1)
    recalled = store.recall(completed, k=1)[0]
    assert recalled.value.provenance == "full"
    assert recalled.score > base.score


def test_hopfield_completion_improves_similarity() -> None:
    """Hopfield densification yields tokens closer to stored keys."""

    full = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    noisy = np.array([0.5, 0.5, 0.0, 0.0], dtype="float32")

    class DummyStore:
        dim = 4

        def recall(self, query, k):
            return [type("T", (), {"key": query})()]

        def to_dense(self, key):
            return key

        def complete(self, query, k=1):
            return full

    store = DummyStore()
    hidden = torch.from_numpy(noisy).view(1, 1, -1)

    spec = TraceSpec(source="episodic", k=1, params={"hopfield": True})
    mem = episodic_retrieve_and_pack(hidden, spec, store, torch.nn.Identity())

    spec2 = TraceSpec(source="episodic", k=1, params={"hopfield": False})
    mem2 = episodic_retrieve_and_pack(hidden, spec2, store, torch.nn.Identity())

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    cos_completed = _cos(mem.tokens[0, 0].numpy(), full)
    cos_raw = _cos(mem2.tokens[0, 0].numpy(), full)
    assert cos_completed > cos_raw
    assert not np.allclose(mem.tokens[0, 0].numpy(), mem2.tokens[0, 0].numpy())


def test_gating_threshold_and_pin_weight() -> None:
    """Pin adds ``delta`` to the score but still requires threshold crossing."""

    store = EpisodicStore(dim=4)
    key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    store.write(key, TraceValue(provenance="a"))

    prob = 1.0  # no surprise, novelty is 0
    query = key

    gate = WriteGate(tau=0.5, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
    decision = gate(prob, query, store.keys(), pin=True)
    assert decision.action != "insert" and decision.score == pytest.approx(0.0)

    strong_gate = WriteGate(tau=0.5, alpha=0.0, beta=0.0, gamma=0.0, delta=1.0)
    decision2 = strong_gate(prob, query, store.keys(), pin=True)
    assert decision2.action == "insert" and decision2.score == pytest.approx(1.0)


def test_low_prob_surprise_crosses_threshold() -> None:
    """Low probability alone can exceed the threshold."""

    gate = WriteGate(tau=1.0, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    query = np.zeros(4, dtype="float32")
    keys = np.zeros((0, 4), dtype="float32")

    high_prob = 0.9
    decision_high = gate(high_prob, query, keys)
    expected_high = surprise(high_prob)
    assert decision_high.action != "insert" and decision_high.score == pytest.approx(expected_high)

    low_prob = 0.1
    decision_low = gate(low_prob, query, keys)
    expected_low = surprise(low_prob)
    assert decision_low.action == "insert" and decision_low.score == pytest.approx(expected_low)


def test_novel_query_crosses_threshold() -> None:
    """Novelty alone can exceed the threshold."""

    gate = WriteGate(tau=0.5, alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")

    seen_keys = np.array([query])
    decision_seen = gate(1.0, query, seen_keys)
    expected_seen = cosine_dissimilarity(query, seen_keys, "max")
    assert decision_seen.action != "insert" and decision_seen.score == pytest.approx(expected_seen)

    novel_keys = np.zeros((0, 4), dtype="float32")
    decision_novel = gate(1.0, query, novel_keys)
    expected_novel = cosine_dissimilarity(query, novel_keys, "max")
    assert decision_novel.action == "insert" and decision_novel.score == pytest.approx(
        expected_novel
    )


def test_writegate_scores_lower_for_duplicate_query() -> None:
    """Duplicate queries receive lower scores than orthogonal ones."""

    gate = WriteGate(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)
    key = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    keys = np.array([key])

    identical = key
    orth = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")

    score_identical = gate.score(1.0, identical, keys)
    score_orth = gate.score(1.0, orth, keys)

    assert score_identical == pytest.approx(0.0)
    assert score_orth == pytest.approx(1.0)
    assert score_identical < score_orth


def test_reward_flag_crosses_threshold() -> None:
    """Reward alone can exceed the threshold."""

    gate = WriteGate(tau=0.5, alpha=0.0, beta=0.0, gamma=1.0, delta=0.0)
    query = np.zeros(4, dtype="float32")
    keys = np.zeros((0, 4), dtype="float32")

    decision_no_reward = gate(1.0, query, keys, reward=0.0)
    expected_none = gate.gamma * 0.0
    assert decision_no_reward.action != "insert" and decision_no_reward.score == pytest.approx(
        expected_none
    )

    decision_reward = gate(1.0, query, keys, reward=1.0)
    expected_reward = gate.gamma * 1.0
    assert decision_reward.action == "insert" and decision_reward.score == pytest.approx(
        expected_reward
    )


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
    memory = MemoryTokens(tokens=traces, mask=torch.ones(1, 4, dtype=torch.bool))

    called = {}
    orig = torch.nn.functional.scaled_dot_product_attention

    def fake_sdp_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        called["hit"] = True
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    monkeypatch.setattr(
        "hippo_mem.common.attn_adapter.F.scaled_dot_product_attention", fake_sdp_attention
    )

    out = adapter(hidden, memory)
    assert called.get("hit")
    assert out.shape == hidden.shape


def test_expand_kv_grouped_query() -> None:
    """Grouped-query attention duplicates K/V heads correctly."""

    cfg = AdapterConfig(hidden_size=8, num_heads=4, num_kv_heads=2, enabled=True)
    adapter = EpisodicAdapter(cfg)
    x = torch.tensor([[[[1.0]], [[2.0]]]])
    expanded = adapter._expand_kv(x)
    assert expanded.shape == (1, 4, 1, 1)
    assert torch.equal(expanded[0, 0], x[0, 0])
    assert torch.equal(expanded[0, 1], x[0, 0])
    assert torch.equal(expanded[0, 2], x[0, 1])
    assert torch.equal(expanded[0, 3], x[0, 1])


def test_expand_kv_multi_query() -> None:
    """Multi-query attention duplicates K/V heads when ``num_kv_heads=1``."""

    cfg = AdapterConfig(hidden_size=8, num_heads=4, num_kv_heads=1, enabled=True)
    adapter = EpisodicAdapter(cfg)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (b=1, kvh=1, t=2, d=2)
    expanded = adapter._expand_kv(x)
    assert expanded.shape == (1, 4, 2, 2)
    for h in range(adapter.num_heads):
        assert torch.equal(expanded[0, h], x[0, 0])


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


def test_replay_queue_score_recency_priority() -> None:
    """Combined weights select highest-priority item."""

    key = np.array([1.0, 0.0], dtype="float32")

    q = ReplayQueue(lambda1=0.6, lambda2=0.4, lambda3=0.0)
    q.add("old_high", key, score=1.0)
    q.add("new_low", key, score=0.5)
    assert q.sample(1)[0] == "old_high"

    q2 = ReplayQueue(lambda1=0.2, lambda2=0.8, lambda3=0.0)
    q2.add("old_high", key, score=1.0)
    q2.add("new_low", key, score=0.5)
    assert q2.sample(1)[0] == "new_low"


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
    assert np.array_equal(np.sort(key.indices), np.sort(key2.indices))
    assert np.allclose(key.values, key2.values)


def test_k_wta_returns_empty_for_non_positive_k() -> None:
    """Non-positive ``k`` yields an empty key."""

    q = np.array([0.2, -0.3], dtype=np.float32)
    for invalid_k in (0, -1):
        key = k_wta(q, invalid_k)
        assert key.indices.size == 0
        assert key.values.size == 0
        assert key.dim == q.size


def test_kwta_produces_sparse_indices() -> None:
    """k-WTA encoding within ``write`` stores only the top-k indices."""

    vec = np.array([0.1, 0.9, -0.8, 0.2], dtype="float32")
    store = EpisodicStore(dim=4, k_wta=2)
    store.write(vec, TraceValue(provenance="x"))
    trace = store.recall(vec, k=1)[0]
    assert trace.key.indices.size == 2
    top2 = np.argsort(-np.abs(vec))[:2]
    assert set(trace.key.indices.tolist()) == set(top2.tolist())


def test_stop_background_tasks_idempotent() -> None:
    """Background maintenance thread can be stopped multiple times."""

    store = EpisodicStore(dim=2, config={"decay_rate": 0.1})
    store.start_background_tasks(interval=0.01)
    store.write(np.ones(2, dtype="float32"), TraceValue(provenance="x"))
    time.sleep(0.02)
    store.stop_background_tasks()
    store.stop_background_tasks()


def test_sparse_query_retrieves_correct_trace() -> None:
    """Sparse querying focuses on top-k indices and ignores distractors."""

    target = np.array([1.0, 1.0, 0.0, 0.0], dtype="float32")
    distractor = np.array([1.0, 0.0, 0.9, 0.0], dtype="float32")
    query = np.array([1.0, 0.901, 0.9, 0.0], dtype="float32")

    dense = EpisodicStore(dim=4)
    dense.write(target, TraceValue(provenance="target"))
    dense.write(distractor, TraceValue(provenance="distractor"))
    dense_res = dense.recall(query, k=1)
    assert dense_res and dense_res[0].value.provenance == "distractor"

    sparse = EpisodicStore(dim=4, k_wta=2)
    sparse.write(target, TraceValue(provenance="target"))
    sparse.write(distractor, TraceValue(provenance="distractor"))
    cue = sparse.to_dense(sparse.sparse_encode(query, sparse.k_wta))
    sparse_res = sparse.recall(cue, k=1)
    assert sparse_res and sparse_res[0].value.provenance == "target"
