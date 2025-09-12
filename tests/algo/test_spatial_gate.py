# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Tests for ``SpatialGate`` heuristics."""

import pytest

from hippo_mem.common.telemetry import gate_registry
from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.map import PlaceGraph
from hippo_mem.training.lora import ingest_spatial_traces


def test_spatial_gate_reduces_repeats() -> None:
    trace = ["P", "Q"] * 5
    g0 = PlaceGraph()
    for ctx in trace:
        g0.observe(ctx)
    writes_plain = g0._log["writes"]

    gate = SpatialGate(block_threshold=1.0, repeat_N=3, recent_window=20, max_degree=64)
    g1 = PlaceGraph()
    prev = None
    for ctx in trace:
        decision = gate.decide(prev, ctx, g1)
        if decision.action == "insert":
            g1.observe(ctx)
        elif decision.action == "aggregate" and prev is not None:
            g1.aggregate_duplicate(prev, ctx)
        prev = ctx
    writes_gate = g1._log["writes"]

    edge = g1.graph[g1._context_to_id["P"]][g1._context_to_id["Q"]]
    assert edge.weight > 1.0
    assert writes_gate < writes_plain


def test_spatial_gate_rejects_bad_config() -> None:
    with pytest.raises(ValueError) as exc:
        SpatialGate(block_threshold=-0.1)
    assert "block_threshold" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        SpatialGate(repeat_N=0)
    assert "repeat_N" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        SpatialGate(max_degree=0)
    assert "max_degree" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        SpatialGate(recent_window=0)
    assert "recent_window" in str(exc.value)


def test_spatial_gate_penalizes_high_degree() -> None:
    gate = SpatialGate(block_threshold=0.3, max_degree=2)
    g = PlaceGraph()
    g.connect("A", "X")
    g.connect("A", "Y")
    g.connect("A", "Z")
    decision = gate.decide(None, "A", g)
    assert decision.action == "route_to_episodic"


def test_spatial_gate_threshold_and_counters() -> None:
    gate_registry.reset()
    graph = PlaceGraph()
    graph.connect("hub", "a")
    graph.connect("hub", "b")
    graph.connect("hub", "c")
    gate = SpatialGate(block_threshold=0.5, max_degree=2)
    decision = SpatialGate(block_threshold=0.5, max_degree=2).decide(None, "hub", graph)
    assert decision.action == "route_to_episodic"
    gate_registry.reset()
    records = [{"trajectory": ["hub"]}, {"trajectory": ["new"]}]
    ingest_spatial_traces(records, graph, gate)
    stats = gate_registry.get("spatial")
    assert stats.attempts == 2
    assert stats.blocked_new_edges == 1
    assert stats.inserted == 1
