import numpy as np
import torch
from torch import nn

from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.relational.adapter import RelationalAdapter
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.retrieval import relational_retrieve_and_pack


def test_relational_retrieve_and_pack_shapes():
    kg = KnowledgeGraph()
    kg.upsert("A", "r", "B", "ctx", head_embedding=[1, 0, 0], tail_embedding=[0, 1, 0])
    kg.upsert("C", "r", "C", "ctx", head_embedding=[0, 0, 1], tail_embedding=[0, 0, 1])

    batch_hidden = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
    spec = TraceSpec(source="relational", k=2, params={"hops": 1})
    proj = nn.Identity()

    mem = relational_retrieve_and_pack(batch_hidden, spec, kg, proj)

    assert mem.tokens.shape == (2, 2, 3)
    assert mem.mask.dtype == torch.bool
    assert mem.mask.tolist() == [[True, True], [True, False]]

    expected_tokens = torch.tensor(
        [
            [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    assert torch.allclose(mem.tokens, expected_tokens)
    assert mem.meta["source"] == "relational"
    assert mem.meta["k"] == 2
    assert mem.meta["hops"] == 1


def test_memory_adapter_matches_direct_adapter():
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
    out = adapter(hidden, memory=mem)

    direct = RelationalAdapter()
    for b in range(hidden.size(0)):
        for t in range(hidden.size(1)):
            feats = tokens[b, mask[b]].numpy()
            fused = direct(hidden[b, t].numpy(), feats)
            fused_t = torch.from_numpy(fused).to(hidden)
            assert torch.allclose(hidden[b, t] + out[b, t], fused_t)
    assert out.shape == hidden.shape


def test_memory_adapter_zero_without_tokens():
    hidden = torch.randn(1, 2, 3)
    tokens = torch.ones(1, 1, 3)
    mask = torch.zeros(1, 1, dtype=torch.bool)
    mem = MemoryTokens(tokens=tokens, mask=mask, meta={"source": "relational"})
    adapter = RelationalMemoryAdapter()
    out = adapter(hidden, memory=mem)
    assert torch.equal(out, torch.zeros_like(hidden))


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
