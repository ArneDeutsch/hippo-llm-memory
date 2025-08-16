"""Tests for relational tuple extraction, KG retrieval and adapter fusion."""

import numpy as np
import pytest

from hippo_mem.relational.adapter import RelationalAdapter
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.tuples import extract_tuples


def test_tuple_precision() -> None:
    """Extractor keeps high-confidence tuples with >=0.9 precision."""

    text = "Alice likes Bob. Bob visited Paris in 2020. Random words. Carol built rockets in 1969."
    tuples = extract_tuples(text, threshold=0.5)

    gold = {
        ("Alice", "likes", "Bob"),
        ("Bob", "visited", "Paris"),
        ("Carol", "built", "rockets"),
    }
    preds = {(h, r, t) for h, r, t, *_ in tuples}
    correct = preds & gold
    precision = len(correct) / len(preds)
    assert precision >= 0.9


def test_multi_hop_retrieval() -> None:
    """Retrieval returns a radius-two subgraph around the best match."""

    kg = KnowledgeGraph()
    kg.upsert("A", "rel", "B", "A to B", head_embedding=[1.0, 0.0])
    kg.upsert("B", "rel", "C", "B to C", head_embedding=[0.5, 0.5])
    kg.upsert("noise", "rel", "D", "noise edge", head_embedding=[0.0, 1.0])

    sub = kg.retrieve([1.0, 0.0], k=1, radius=2)
    assert set(sub.nodes()) == {"A", "B", "C"}


def test_dual_path_fusion_deterministic() -> None:
    """Cross-attention fusion is deterministic for a toy example."""

    adapter = RelationalAdapter()
    query = np.array([1.0, 0.0])
    kg_feats = np.array([[1.0, 0.0], [0.0, 1.0]])
    epi_feats = np.array([[0.0, 1.0]])
    out1 = adapter(query, kg_feats, epi_feats, kg_conf=0.8, episodic_conf=0.2)
    out2 = adapter(query, kg_feats, epi_feats, kg_conf=0.8, episodic_conf=0.2)

    kg_attn = np.array([np.exp(1) / (np.exp(1) + 1), 1 / (np.exp(1) + 1)])
    epi_attn = np.array([0.0, 1.0])
    expected = 0.8 * kg_attn + 0.2 * epi_attn

    assert np.allclose(out1, out2)
    assert np.allclose(out1, expected)


@pytest.mark.parametrize("threshold, expect_low", [(0.8, False), (0.2, True)])
def test_schema_threshold_routes_confident_tuples(threshold: float, expect_low: bool) -> None:
    """High schema threshold keeps low-confidence tuples out of the KG."""

    kg = KnowledgeGraph(config={"schema_threshold": threshold})
    kg.schema_index.add_schema("likes", "likes")

    high = ("Alice", "likes", "Bob", "ctx", None, 0.9, 0)
    low = ("Carol", "likes", "Dave", "ctx", None, 0.5, 1)

    kg.ingest(high)
    kg.ingest(low)

    assert kg.graph.has_edge("Alice", "Bob")
    assert kg.graph.has_edge("Carol", "Dave") == expect_low
    if expect_low:
        assert kg.schema_index.episodic_buffer == []
    else:
        assert kg.schema_index.episodic_buffer == [low]
