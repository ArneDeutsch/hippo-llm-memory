"""Tests for spatial mapping and macros.

These tests exercise the context→place encoder stub, the topological
graph with A*/Dijkstra path finding, and the minimal macro library.
"""

from hippo_mem.spatial.macros import MacroLibrary, plan_route
from hippo_mem.spatial.map import SpatialMap


def test_context_encoder_is_deterministic() -> None:
    """Encoding the same context twice yields the same coordinates."""

    world = SpatialMap()
    a = world.add_place("home")
    b = world.add_place("home")
    assert a.coord == b.coord


def test_astar_matches_dijkstra() -> None:
    """When edge weights are based on geometry, A* equals Dijkstra."""

    world = SpatialMap()
    world.connect("A", "B")
    world.connect("B", "C")
    world.connect("A", "C")
    assert world.shortest_path("A", "C") == world.dijkstra("A", "C")


def test_plan_route_with_custom_weights() -> None:
    """Dijkstra-based planning respects explicit edge weights."""

    world = SpatialMap()
    world.connect("A", "B", weight=1)
    world.connect("B", "C", weight=1)
    world.connect("A", "C", weight=5)
    assert plan_route(world, "A", "C") == ["A", "B", "C"]


def test_behavior_cloning_macro_library() -> None:
    """The macro library stores the shortest demonstration."""

    demos = [["A", "B", "C"], ["A", "C"]]
    library = MacroLibrary()
    macro = library.behavior_clone("go", demos)
    assert macro.trajectory == ["A", "C"]
    assert library.get("go").trajectory == ["A", "C"]
