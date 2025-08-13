"""Tests for relational tuple extraction and knowledge graph utilities."""

import numpy as np

from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.tuples import extract_tuples


def test_tuple_extraction() -> None:
    """The tuple extractor returns the expected structure."""

    text = "Alice met Bob in 2020."
    tuples = extract_tuples(text)
    assert tuples == [("Alice", "met Bob", "Alice met Bob in 2020", "2020")]


def test_kg_subgraph_retrieval() -> None:
    """Subgraphs are retrieved around the most similar entity."""

    text = "Alice met Bob in 2020."
    tuples = extract_tuples(text)

    kg = KnowledgeGraph()
    for tup in tuples:
        kg.add_tuple(*tup)

    # Assign embeddings to two nodes; the query is closest to "Alice".
    kg.set_embedding("Alice", np.array([1.0, 0.0]))
    kg.set_embedding("noise", np.array([0.0, 1.0]))

    sub = kg.subgraph(np.array([1.0, 0.0]), k=1, radius=1)
    assert "Alice" in sub.nodes
    # The relation node should also be present via the edge from Alice.
    assert ("Alice", "met Bob") in sub.edges
