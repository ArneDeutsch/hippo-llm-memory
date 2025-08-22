"""Tests for ``RelationalGate`` salience gating."""

from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph


def test_relational_gate_skips_duplicates() -> None:
    gate = RelationalGate()
    kg = KnowledgeGraph(config={"schema_threshold": 0.0}, gate=gate)
    kg.schema_index.add_schema("likes", "likes")
    tup = ("A", "likes", "B", "ctx", None, 0.9, 0)

    assert kg.ingest(tup)[0] == "insert"
    assert kg.ingest(tup)[0] == "aggregate"
    with_gate = kg.graph.number_of_edges()

    kg2 = KnowledgeGraph(config={"schema_threshold": 0.0})
    kg2.schema_index.add_schema("likes", "likes")
    kg2.ingest(tup)
    kg2.ingest(tup)
    without_gate = kg2.graph.number_of_edges()

    edge = next(iter(kg.graph.get_edge_data("A", "B").values()))
    assert edge["conf"] > 0.9
    assert with_gate < without_gate
