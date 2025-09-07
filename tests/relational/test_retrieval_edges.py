import types

import networkx as nx
import numpy as np
import pytest
import torch
from torch import nn

from hippo_mem.common import TraceSpec
from hippo_mem.relational.retrieval import NodePooler, relational_retrieve_and_pack


def test_node_pooler_self_policy() -> None:
    kg = types.SimpleNamespace(node_embeddings={"n": np.array([1.0, 2.0])})
    pooler = NodePooler(policy="self")
    sub = nx.MultiDiGraph()
    out = pooler(kg, sub, "n", 2)
    assert np.array_equal(out, np.array([1.0, 2.0]))


def test_node_pooler_unknown_policy() -> None:
    kg = types.SimpleNamespace(node_embeddings={})
    pooler = NodePooler(policy="bogus")
    sub = nx.MultiDiGraph()
    with pytest.raises(ValueError):
        pooler(kg, sub, "n", 2)


def test_relational_retrieve_limit_zero_empty() -> None:
    kg = types.SimpleNamespace(
        node_embeddings={}, retrieve=lambda *a, **k: nx.MultiDiGraph(), dim=2
    )
    hidden = torch.zeros(1, 1, 2)
    spec = TraceSpec(source="test", k=0, params={"limit": 0})
    mem = relational_retrieve_and_pack(hidden, spec, kg, nn.Identity())
    assert mem.tokens.shape[1] == 0
    assert mem.mask.shape[1] == 0
