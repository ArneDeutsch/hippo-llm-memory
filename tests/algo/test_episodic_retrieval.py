# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import numpy as np
import pytest
import torch
from torch import nn

from hippo_mem.common import TraceSpec
from hippo_mem.common.telemetry import registry
from hippo_mem.episodic.retrieval import episodic_retrieve_and_pack


class DummyStore:
    dim = 4

    def __init__(self):
        self.calls = 0

    def recall(self, query, k):  # pragma: no cover - simple stub
        self.calls += 1
        if self.calls == 1:
            vecs = [np.ones(self.dim, dtype=np.float32), np.ones(self.dim, dtype=np.float32) * 2]
        else:
            vecs = [np.ones(self.dim, dtype=np.float32) * 3]
        return [type("T", (), {"key": v})() for v in vecs]

    def to_dense(self, key):  # pragma: no cover - simple stub
        return key


def test_retrieve_and_pack_shapes():
    batch_hidden = torch.zeros(2, 3, 4)
    spec = TraceSpec(source="episodic", k=2)
    store = DummyStore()
    proj = nn.Linear(store.dim, 4)
    mem = episodic_retrieve_and_pack(batch_hidden, spec, store, proj)
    assert mem.tokens.shape == (2, 2, 4)
    assert mem.mask.dtype == torch.bool
    assert mem.mask.tolist() == [[True, True], [True, False]]
    assert mem.meta["k"] == 2
    assert mem.meta["latency_ms"] >= 0
    assert mem.meta["hit_rate"] == pytest.approx(0.75)


def test_retrieve_with_sparse_encoding():
    class SparseStore(DummyStore):
        k_wta = 1

        def __init__(self):
            super().__init__()
            self.sparse_called = False
            self.dense_called = False

        def sparse_encode(self, cue, k):
            self.sparse_called = True
            return cue + 1

        def to_dense(self, key):
            self.dense_called = True
            return key

        def recall(self, query, k):
            self.recall_cue = query
            return [type("T", (), {"key": np.ones(self.dim, dtype=np.float32)})()]

    store = SparseStore()
    batch_hidden = torch.zeros(1, 1, 4)
    spec = TraceSpec(source="episodic", k=1)
    mem = episodic_retrieve_and_pack(batch_hidden, spec, store, nn.Identity())
    assert store.sparse_called
    assert store.dense_called
    assert np.array_equal(store.recall_cue, np.ones(store.dim, dtype=np.float32))
    assert mem.mask.tolist() == [[True]]


def test_hopfield_fills_on_empty_recall():
    class HopfieldStore(DummyStore):
        def __init__(self):
            super().__init__()
            self.complete_called = False

        def recall(self, query, k):
            return []

        def complete(self, cue, k):
            self.complete_called = True
            return cue + 5

    store = HopfieldStore()
    batch_hidden = torch.zeros(1, 1, 4)
    spec = TraceSpec(source="episodic", k=1)
    mem = episodic_retrieve_and_pack(batch_hidden, spec, store, nn.Identity())
    assert store.complete_called
    assert mem.mask.tolist() == [[True]]
    assert torch.allclose(mem.tokens[0, 0], torch.full((4,), 5.0))


def test_k_zero_returns_empty_tokens():
    class NoCallStore(DummyStore):
        def recall(self, query, k):  # pragma: no cover - should not be called
            raise AssertionError("recall should not be called when k=0")

    store = NoCallStore()
    batch_hidden = torch.zeros(2, 3, 4)
    spec = TraceSpec(source="episodic", k=0)
    mem = episodic_retrieve_and_pack(batch_hidden, spec, store, nn.Identity())
    assert mem.tokens.shape == (2, 0, 4)
    assert mem.mask.shape == (2, 0)
    assert mem.meta["hit_rate"] == 0.0


def test_empty_store_has_zero_hit_rate():
    class EmptyStore(DummyStore):
        def recall(self, query, k):
            return []

        def complete(self, cue, k):
            return cue

    registry.reset()
    store = EmptyStore()
    batch_hidden = torch.zeros(1, 1, 4)
    k = 3
    spec = TraceSpec(source="episodic", k=k)
    episodic_retrieve_and_pack(batch_hidden, spec, store, nn.Identity())
    snap = registry.get("episodic").snapshot()
    assert snap["hits_at_k"] == 0
    assert snap["hit_rate_at_k"] == 0.0
    assert snap["tokens_returned"] == k
