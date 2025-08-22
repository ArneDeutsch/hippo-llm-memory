"""Semantic behaviour tests for ``RelationalGate``."""

import pytest

from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph


def test_relational_gate_semantics() -> None:
    """Duplicate aggregation and episodic routing semantics."""
    # Gate disabled: duplicates insert two edges
    kg_plain = KnowledgeGraph(config={"schema_threshold": 0.0})
    kg_plain.schema_index.add_schema("likes", "likes")
    tup = ("A", "likes", "B", "ctx", None, 1.0, 0)
    kg_plain.ingest(tup)
    kg_plain.ingest(tup)
    assert kg_plain.graph.number_of_edges() == 2

    # Gate enabled: duplicate aggregates and low-score tuple routes to episodic
    gate = RelationalGate(threshold=0.6, w_conf=1.0, w_nov=0.0, w_deg=0.0, w_rec=0.0)
    kg_gate = KnowledgeGraph(config={"schema_threshold": 0.0}, gate=gate)
    kg_gate.schema_index.add_schema("likes", "likes")
    kg_gate.ingest(tup)
    kg_gate.ingest(tup)
    assert kg_gate.graph.number_of_edges() == 1
    edge = next(iter(kg_gate.graph.get_edge_data("A", "B").values()))
    assert edge["conf"] == pytest.approx(2.0)

    low_conf = ("C", "likes", "D", "ctx", None, 0.1, 1)
    action, _ = kg_gate.ingest(low_conf)
    assert action == "route_to_episodic"
    assert len(kg_gate._episodic_queue) == 1
