# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Tests for ``RelationalGate`` salience gating."""

import time

import pytest

from hippo_mem.common.telemetry import gate_registry
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph


def test_relational_gate_skips_duplicates() -> None:
    gate = RelationalGate()
    kg = KnowledgeGraph(config={"schema_threshold": 0.0}, gate=gate)
    kg.schema_index.add_schema("likes", "likes")
    tup = ("A", "likes", "B", "ctx", None, 0.9, 0)

    assert kg.ingest(tup).action == "insert"
    assert kg.ingest(tup).action == "aggregate"
    with_gate = kg.graph.number_of_edges()

    kg2 = KnowledgeGraph(config={"schema_threshold": 0.0})
    kg2.schema_index.add_schema("likes", "likes")
    kg2.ingest(tup)
    kg2.ingest(tup)
    without_gate = kg2.graph.number_of_edges()

    edge = next(iter(kg.graph.get_edge_data("A", "B").values()))
    assert edge["conf"] > 0.9
    assert with_gate < without_gate


def test_relational_gate_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        RelationalGate(threshold=1.1)
    with pytest.raises(ValueError):
        RelationalGate(w_conf=-0.1)
    with pytest.raises(ValueError):
        RelationalGate(max_degree=0)
    with pytest.raises(ValueError):
        RelationalGate(recency_window=0)


def test_relational_gate_penalizes_high_degree() -> None:
    gate = RelationalGate(threshold=0.8, max_degree=1)
    kg = KnowledgeGraph(config={"schema_threshold": 0.0})
    kg.graph.add_edge("A", "X", relation="knows", conf=1.0)
    kg.graph.add_edge("A", "Y", relation="knows", conf=1.0)
    gate._last_seen["A"] = time.time()
    gate._last_seen["B"] = time.time()
    tup = ("A", "likes", "B", "ctx", None, 1.0, 0)
    deg_pen = gate._degree_penalty("A", "B", kg)
    prev_seen = gate._last_seen["A"]
    decision = gate.decide(tup, kg)
    assert decision.action == "route_to_episodic"
    assert deg_pen > 0.0
    assert decision.score is not None and decision.score < gate.threshold
    assert gate._last_seen["A"] > prev_seen


def test_relational_gate_threshold_and_counters() -> None:
    gate_registry.reset()
    gate = RelationalGate(threshold=0.8, w_conf=1.0, w_nov=0.0, w_deg=0.0, w_rec=0.0)
    kg = KnowledgeGraph(config={"schema_threshold": 0.0}, gate=gate)
    t_thr = ("A", "rel", "B", "ctx", None, 0.8, 0)
    t_low = ("C", "rel", "D", "ctx", None, 0.79, 0)
    assert kg.ingest(t_thr).action == "insert"
    assert kg.ingest(t_low).action == "route_to_episodic"
    stats = gate_registry.get("relational")
    assert stats.attempts == 2
    assert stats.inserted == 1
    assert stats.routed_to_episodic == 1
