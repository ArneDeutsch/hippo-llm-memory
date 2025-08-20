import networkx as nx
import numpy as np
import torch
from torch import nn

from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
from hippo_mem.common import TraceSpec
from hippo_mem.relational.retrieval import relational_retrieve_and_pack


class DummyKG:
    def __init__(self):
        self.node_embeddings = {
            "A": np.ones(3, dtype=np.float32),
            "B": np.ones(3, dtype=np.float32) * 2,
            "C": np.ones(3, dtype=np.float32) * 3,
        }
        self.calls = 0

    def retrieve(self, _query, k=1, radius=1):  # pragma: no cover - stub
        self.calls += 1
        g = nx.MultiDiGraph()
        if self.calls == 1:
            g.add_node("A")
            g.add_node("B")
        else:
            g.add_node("C")
        return g

    @property
    def dim(self):  # pragma: no cover - simple property
        return 3


def test_relational_retrieve_and_pack_shapes():
    batch_hidden = torch.zeros(2, 3, 4)
    spec = TraceSpec(source="relational", k=2, params={"hops": 1})
    kg = DummyKG()
    proj = nn.Linear(3, 4)
    mem = relational_retrieve_and_pack(batch_hidden, spec, kg, proj)
    assert mem.tokens.shape == (2, 2, 4)
    assert mem.mask.dtype == torch.bool
    assert mem.mask.tolist() == [[True, True], [True, False]]
    assert mem.meta["source"] == "relational"


def test_adapter_consumes_relational_tokens():
    batch_hidden = torch.zeros(2, 3, 4)
    spec = TraceSpec(source="relational", k=2)
    kg = DummyKG()
    proj = nn.Linear(3, 4)
    mem = relational_retrieve_and_pack(batch_hidden, spec, kg, proj)
    adapter = RelationalMemoryAdapter()
    out = adapter(batch_hidden, memory=mem)
    assert out.abs().sum() > 0
