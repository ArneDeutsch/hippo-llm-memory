"""Tests for spatial memory stubs."""

from hippo_mem.spatial.macros import plan_route
from hippo_mem.spatial.map import SpatialMap


def test_spatial_map() -> None:
    """Points are added and routes planned."""
    world = SpatialMap()
    world.add_point(0.0, 0.0)
    assert plan_route("A", "B") == ["A", "B"]
