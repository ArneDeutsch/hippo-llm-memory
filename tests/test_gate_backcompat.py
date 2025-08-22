"""Back-compat regression test when gates are disabled."""

from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.map import PlaceGraph


def test_gate_backcompat_counts() -> None:
    """Node and edge counts match pre-gate baseline when gates are off."""
    # Relational graph baseline
    kg = KnowledgeGraph(config={"schema_threshold": 0.0})
    kg.schema_index.add_schema("likes", "likes")
    tuples = [
        ("A", "likes", "B", "c1", None, 0.9, 0),
        ("A", "likes", "B", "c2", None, 0.8, 1),
        ("C", "likes", "D", "c3", None, 0.7, 2),
    ]
    for tup in tuples:
        kg.ingest(tup)
    assert kg.graph.number_of_nodes() == 4
    assert kg.graph.number_of_edges() == 3

    # Spatial map baseline
    g = PlaceGraph()
    for ctx in ["A", "B", "C", "A"]:
        g.observe(ctx)
    assert len(g.graph) == 3
    edge_count = sum(len(nbrs) for nbrs in g.graph.values())
    assert edge_count == 6
