"""Tests for ``SpatialGate`` heuristics."""

import pytest

from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.map import PlaceGraph


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
        action, _ = gate.decide(prev, ctx, g1)
        if action == "insert":
            g1.observe(ctx)
        elif action == "aggregate" and prev is not None:
            g1.aggregate_duplicate(prev, ctx)
        prev = ctx
    writes_gate = g1._log["writes"]

    edge = g1.graph[g1._context_to_id["P"]][g1._context_to_id["Q"]]
    assert edge.weight > 1.0
    assert writes_gate < writes_plain


def test_spatial_gate_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        SpatialGate(block_threshold=1.5)
    with pytest.raises(ValueError):
        SpatialGate(repeat_N=0)
    with pytest.raises(ValueError):
        SpatialGate(max_degree=0)
