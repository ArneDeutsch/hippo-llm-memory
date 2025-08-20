import numpy as np
import pytest
import torch
from torch import nn

from hippo_mem.common import TraceSpec
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
