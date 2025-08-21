import networkx as nx
import numpy as np
import torch
from torch import nn

from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
from hippo_mem.common import TraceSpec
from hippo_mem.relational.kg import KnowledgeGraph
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


def test_relational_hops_two_edges():
    kg = KnowledgeGraph()
    kg.upsert("A", "r", "B", "ctx", head_embedding=[1, 0, 0], tail_embedding=[0, 1, 0])
    kg.upsert("B", "r", "C", "ctx", head_embedding=[0, 1, 0], tail_embedding=[0, 0, 1])
    batch_hidden = torch.tensor([[[1.0, 0.0, 0.0]]])
    proj = nn.Identity()
    c_vec = torch.tensor([0.0, 0.0, 1.0], dtype=batch_hidden.dtype)

    spec1 = TraceSpec(source="relational", k=3, params={"hops": 1})
    mem1 = relational_retrieve_and_pack(batch_hidden, spec1, kg, proj)
    assert mem1.mask.tolist() == [[True, True, False]]
    assert not any(torch.allclose(tok, c_vec) for tok in mem1.tokens[0][mem1.mask[0]])

    spec2 = TraceSpec(source="relational", k=3, params={"hops": 2})
    mem2 = relational_retrieve_and_pack(batch_hidden, spec2, kg, proj)
    assert mem2.mask.tolist() == [[True, True, True]]
    assert any(torch.allclose(tok, c_vec) for tok in mem2.tokens[0][mem2.mask[0]])
