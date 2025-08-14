"""Tests for relational tuple extraction, KG retrieval and adapter fusion."""

import numpy as np

from hippo_mem.relational.adapter import RelationalAdapter
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.tuples import extract_tuples


def test_tuple_precision() -> None:
    """Extractor keeps high-confidence tuples with >=0.9 precision."""

    text = (
        "Alice likes Bob. Bob visited Paris in 2020. Random words. " "Carol built rockets in 1969."
    )
    tuples = extract_tuples(text, threshold=0.5)

    gold = {
        ("Alice", "likes Bob", "Alice likes Bob", None),
        ("Bob", "visited Paris", "Bob visited Paris in 2020", "2020"),
        ("Carol", "built rockets", "Carol built rockets in 1969", "1969"),
    }
    correct = [t[:4] for t in tuples if t[:4] in gold]
    precision = len(correct) / len(tuples)
    assert precision >= 0.9


def test_multi_hop_retrieval() -> None:
    """Retrieval returns a radius-two subgraph around the best match."""

    kg = KnowledgeGraph()
    kg.upsert("A", "B", "A to B", entity_embedding=[1.0, 0.0])
    kg.upsert("B", "C", "B to C", entity_embedding=[0.5, 0.5])
    kg.upsert("noise", "D", "noise edge", entity_embedding=[0.0, 1.0])

    sub = kg.retrieve([1.0, 0.0], k=1, radius=2)
    assert set(sub.nodes()) == {"A", "B", "C"}


def test_dual_path_fusion_deterministic() -> None:
    """Cross-attention fusion is deterministic for a toy example."""

    adapter = RelationalAdapter()
    query = np.array([1.0, 0.0])
    feats = np.array([[1.0, 0.0], [0.0, 1.0]])
    out1 = adapter(query, feats)
    out2 = adapter(query, feats)
    expected = np.array([np.exp(1) / (np.exp(1) + 1), 1 / (np.exp(1) + 1)])

    assert np.allclose(out1, out2)
    assert np.allclose(out1, expected)
