# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import networkx as nx
import numpy as np
import torch
from torch import nn

from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.relational.adapter import RelationalAdapter
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.retrieval import relational_retrieve_and_pack


class DummyKG:
    """Minimal KG returning deterministic subgraphs."""

    def __init__(self) -> None:
        self.node_embeddings = {
            "A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "B": np.array([0.0, 2.0, 0.0], dtype=np.float32),
            "C": np.array([0.0, 0.0, 3.0], dtype=np.float32),
        }
        self.calls = 0

    def retrieve(self, _query, k=1, radius=1):  # pragma: no cover - stub
        self.calls += 1
        g = nx.MultiDiGraph()
        if self.calls == 1:
            g.add_edge("A", "B")  # A has B as neighbor
        else:
            g.add_node("C")
        return g

    @property
    def dim(self):  # pragma: no cover - simple property
        return 3


def test_relational_retrieve_and_pack_tokens_and_meta():
    batch_hidden = torch.zeros(2, 1, 3)
    spec = TraceSpec(source="relational", k=2, params={"hops": 1})
    kg = DummyKG()
    proj = nn.Identity()
    mem = relational_retrieve_and_pack(batch_hidden, spec, kg, proj)

    expected = torch.tensor(
        [
            [[0.5, 1.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
        ],
        dtype=batch_hidden.dtype,
    )
    assert torch.allclose(mem.tokens, expected)
    assert mem.tokens.shape == (2, 2, 3)
    assert mem.mask.tolist() == [[True, True], [True, False]]
    assert mem.meta["source"] == "relational"
    assert mem.meta["k"] == 2
    assert mem.meta["hops"] == 1


def test_adapter_consumes_relational_tokens() -> None:
    """Memory adapter matches direct fusion for a known batch."""

    kg = KnowledgeGraph()
    kg.node_embeddings["A"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    kg.node_embeddings["B"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    hidden = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        ]
    )
    tokens = torch.from_numpy(
        np.stack([kg.node_embeddings["A"], kg.node_embeddings["B"]])
    ).unsqueeze(1)
    mask = torch.tensor([[True], [True]])
    mem = MemoryTokens(tokens=tokens, mask=mask, meta={"source": "relational"})

    adapter = RelationalMemoryAdapter()
    residual = adapter(hidden, memory=mem)

    direct = RelationalAdapter()
    expected = torch.empty_like(hidden)
    for b in range(hidden.size(0)):
        feats = tokens[b, mask[b]].numpy()
        for t in range(hidden.size(1)):
            expected[b, t] = torch.from_numpy(direct(hidden[b, t].numpy(), feats))

    assert torch.allclose(hidden + residual, expected)
    assert residual.shape == hidden.shape


def test_adapter_zero_without_tokens() -> None:
    """Empty masks produce a zero residual."""

    hidden = torch.randn(1, 2, 3)
    tokens = torch.ones(1, 1, 3)
    mask = torch.zeros(1, 1, dtype=torch.bool)
    mem = MemoryTokens(tokens=tokens, mask=mask, meta={"source": "relational"})
    adapter = RelationalMemoryAdapter()
    residual = adapter(hidden, memory=mem)
    assert torch.equal(residual, torch.zeros_like(hidden))
    assert residual.shape == hidden.shape
    assert not mem.mask.any()


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


def test_relational_fallback_on_missing_edges() -> None:
    kg = KnowledgeGraph()
    batch_hidden = torch.zeros(1, 1, 3)
    spec = TraceSpec(source="relational", k=1, params={"hops": 1})
    proj = nn.Identity()
    mem = relational_retrieve_and_pack(batch_hidden, spec, kg, proj)
    assert mem.tokens.shape == (1, 1, 3)
    assert mem.mask.tolist() == [[False]]
    assert mem.meta["hit_rate"] == 0.0
