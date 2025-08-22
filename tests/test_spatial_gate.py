"""Tests for ``SpatialGate`` heuristics."""

from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.map import PlaceGraph


def test_spatial_gate_reduces_repeats() -> None:
    trace = ["P"] * 5 + ["Q"] * 5
    g0 = PlaceGraph()
    for ctx in trace:
        g0.observe(ctx)
    writes_plain = g0._log["writes"]

    gate = SpatialGate(block_threshold=1.0, repeat_N=3, recent_window=20, max_degree=64)
    g1 = PlaceGraph()
    for ctx in trace:
        if gate.allow(ctx, g1):
            g1.observe(ctx)
    writes_gate = g1._log["writes"]

    assert writes_gate < writes_plain
