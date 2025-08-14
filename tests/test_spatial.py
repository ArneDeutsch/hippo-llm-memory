"""Tests for the spatial map and macro library."""

from hippo_mem.spatial.macros import MacroLib
from hippo_mem.spatial.map import PlaceGraph


def _run_observations(seq: list[str]) -> tuple[list[int], PlaceGraph]:
    graph = PlaceGraph()
    ids = [graph.observe(ctx) for ctx in seq]
    return ids, graph


def test_observations_are_deterministic() -> None:
    """Sequences of observations should grow the same graph."""

    seq = ["A", "B", "C", "A", "C", "B"]
    ids1, g1 = _run_observations(seq)
    ids2, g2 = _run_observations(seq)
    assert ids1 == ids2
    assert g1.graph == g2.graph


def test_planner_finds_shortest_path_on_grid() -> None:
    """A small 2×2 grid has two equivalent shortest routes."""

    g = PlaceGraph()
    g.connect("0,0", "1,0")
    g.connect("0,0", "0,1")
    g.connect("1,0", "1,1")
    g.connect("0,1", "1,1")
    path = g.plan("0,0", "1,1")
    assert path in (["0,0", "1,0", "1,1"], ["0,0", "0,1", "1,1"])


def test_planner_shortest_path_with_custom_weights() -> None:
    """A* and Dijkstra agree on a simple weighted triangle."""

    g = PlaceGraph()
    g.connect("A", "B", cost=1)
    g.connect("B", "C", cost=1)
    g.connect("A", "C", cost=5)
    assert g.plan("A", "C", method="dijkstra") == ["A", "B", "C"]


def test_macro_replay_reduces_steps() -> None:
    """Macros should be shorter than a baseline planned route."""

    g = PlaceGraph()
    g.connect("start", "a")
    g.connect("a", "b")
    g.connect("b", "goal")
    baseline = g.plan("start", "goal")

    lib = MacroLib()
    lib.store("shortcut", ["start", "goal"])
    macro = lib.suggest("start", "goal", k=1)[0]
    assert len(macro.trajectory) < len(baseline)
