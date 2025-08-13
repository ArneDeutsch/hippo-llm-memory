"""Tests for relational memory stubs."""

from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.tuples import make_tuple


def test_relational_graph() -> None:
    """Edges are added to the knowledge graph."""
    kg = KnowledgeGraph()
    edge = make_tuple("a", "b")
    kg.add_edge(*edge)
    assert edge[1] in kg.edges.get("a", [])
