from __future__ import annotations

from hippo_mem.memory.spatial_store import SpatialStore, Transition


def test_spatial_store_writes_edges() -> None:
    store = SpatialStore()
    store.write(Transition("A", "B", cost=2.0))
    store.write(Transition("B", "C", cost=1.0))
    a = store.graph._context_to_id["A"]
    b = store.graph._context_to_id["B"]
    c = store.graph._context_to_id["C"]
    assert store.graph.graph[a][b].cost == 2.0
    assert store.graph.graph[b][c].cost == 1.0


def test_spatial_store_plan_uses_astar() -> None:
    store = SpatialStore()
    store.write(Transition("A", "B", cost=1.0))
    store.write(Transition("B", "C", cost=1.0))
    store.write(Transition("A", "C", cost=3.0))
    path = store.plan("A", "C")
    assert path == ["A", "B", "C"]
