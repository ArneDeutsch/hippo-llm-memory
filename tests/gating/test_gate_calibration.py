import numpy as np

from hippo_mem.common.telemetry import gate_registry
from hippo_mem.episodic.gating import WriteGate, gate_batch
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.map import PlaceGraph


def _rate(name: str) -> float:
    stats = gate_registry.get(name)
    assert stats.attempts > 0
    assert 0 < stats.accepted < stats.attempts
    return stats.accepted / stats.attempts


def test_episodic_gate_calibrates() -> None:
    gate_registry.reset()
    gate = WriteGate(tau=0.5)
    probs = np.array([0.9, 0.9, 0.1, 0.1], dtype="float32")
    queries = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype="float32")
    keys = np.array([[1.0, 0.0]], dtype="float32")
    gate_batch(gate, probs, queries, keys)
    rate = _rate("episodic")
    assert 0.5 <= rate <= 0.8


def test_relational_gate_calibrates() -> None:
    gate_registry.reset()
    gate = RelationalGate(threshold=0.6, w_conf=1.0, w_nov=0.0, w_deg=0.0, w_rec=0.0)
    kg = KnowledgeGraph(gate=gate)
    tuples = [
        ("A", "likes", "B", "ctx", "t", 0.9, 0),
        ("A", "likes", "B", "ctx", "t", 0.9, 0),
        ("C", "likes", "D", "ctx", "t", 0.1, 0),
        ("E", "likes", "F", "ctx", "t", 0.9, 0),
    ]
    for tup in tuples:
        kg.ingest(tup)
    rate = _rate("relational")
    assert 0.5 <= rate <= 0.8


def test_spatial_gate_calibrates() -> None:
    gate_registry.reset()
    gate = SpatialGate(block_threshold=0.5)
    graph = PlaceGraph()
    prev = None
    for ctx in ["A", "B", "A", "B"]:
        decision = gate.decide(prev, ctx, graph)
        if decision.action == "insert":
            graph.observe(ctx)
        elif decision.action == "aggregate" and prev is not None:
            graph.aggregate_duplicate(prev, ctx)
        prev = ctx
    rate = _rate("spatial")
    assert 0.5 <= rate <= 0.8
