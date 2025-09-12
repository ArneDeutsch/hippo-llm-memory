# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

from types import SimpleNamespace

from hippo_eval.metrics.scoring import spatial_kpis
from hippo_mem.memory.spatial_store import SpatialStore, Transition
from hippo_mem.planning.path_planner import PathPlanner


def test_path_planner_finds_optimal_path() -> None:
    store = SpatialStore()
    store.write(Transition("S", "A", 1.0))
    store.write(Transition("A", "G", 1.0))
    store.write(Transition("S", "G", 5.0))
    planner = PathPlanner(store.graph)
    assert planner.shortest_path("S", "G") == ["S", "A", "G"]


def test_spatial_kpis_suboptimality_threshold() -> None:
    task = SimpleNamespace(
        prompt="Grid 3x3 with obstacles [] Start (0,0) goal (2,0)",
    )
    rows = [{"pred": "RR"}]
    metrics = spatial_kpis([task], rows)
    assert rows[0]["success"]
    assert metrics["suboptimality_ratio"] == 1.0

    rows = [{"pred": "RDRU"}]
    spatial_kpis([task], rows)
    assert not rows[0]["success"]
