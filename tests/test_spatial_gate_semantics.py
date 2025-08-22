"""Semantic behaviour tests for ``SpatialGate``."""

from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.map import PlaceGraph


def test_spatial_gate_blocks_self_and_flaps() -> None:
    """Gate reduces self-loop writes and aggregates flapping edges."""
    # Self-loop A->A->A
    gate_self = SpatialGate(block_threshold=1.0, repeat_N=3, recent_window=20, max_degree=64)
    trace_self = ["A", "A", "A"]
    g_plain = PlaceGraph()
    for ctx in trace_self:
        g_plain.observe(ctx)
    writes_plain = g_plain._log["writes"]

    g_gate = PlaceGraph()
    prev = None
    for ctx in trace_self:
        action, _ = gate_self.decide(prev, ctx, g_gate)
        if action == "insert":
            g_gate.observe(ctx)
        elif action == "aggregate":
            g_gate.aggregate_duplicate(prev, ctx)
        prev = ctx
    assert g_gate._log["writes"] == 2
    assert writes_plain == 3

    # Flapping A<->B
    gate_flap = SpatialGate(block_threshold=1.0, repeat_N=3, recent_window=20, max_degree=64)
    trace_flap = ["A", "B", "A", "B"]
    g_plain2 = PlaceGraph()
    for ctx in trace_flap:
        g_plain2.observe(ctx)
    writes_plain2 = g_plain2._log["writes"]
    edge_plain = g_plain2.graph[g_plain2._context_to_id["A"]][g_plain2._context_to_id["B"]]
    assert edge_plain.weight == 1.0

    g_gate2 = PlaceGraph()
    prev = None
    for ctx in trace_flap:
        action, _ = gate_flap.decide(prev, ctx, g_gate2)
        if action == "insert":
            g_gate2.observe(ctx)
        elif action == "aggregate":
            g_gate2.aggregate_duplicate(prev, ctx)
        prev = ctx
    writes_gate2 = g_gate2._log["writes"]
    edge_gate = g_gate2.graph[g_gate2._context_to_id["A"]][g_gate2._context_to_id["B"]]
    assert edge_gate.weight > 1.0
    assert writes_gate2 == 2
    assert writes_plain2 == 4
