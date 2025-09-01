from __future__ import annotations

from dataclasses import dataclass
from typing import List

from hippo_mem.planning.path_planner import PathPlanner
from hippo_mem.spatial.place_graph import PlaceGraph


@dataclass
class Transition:
    """Directed edge between two contexts.

    Parameters
    ----------
    src:
        Source context name.
    dst:
        Destination context name.
    cost:
        Transition cost; defaults to ``1.0``.
    """

    src: str
    dst: str
    cost: float = 1.0


class SpatialStore:
    """Lightweight store backing the spatial memory."""

    def __init__(self) -> None:
        self.graph = PlaceGraph()
        self.planner = PathPlanner(self.graph)

    def write(self, transition: Transition) -> None:
        """Insert ``transition`` into the map."""

        self.graph.observe(transition.src)
        self.graph.observe(transition.dst)
        self.graph.connect(transition.src, transition.dst, cost=transition.cost)

    def plan(self, start: str, goal: str) -> List[str]:
        """Return planned path between ``start`` and ``goal`` using A* search."""

        return self.planner.shortest_path(start, goal)


__all__ = ["SpatialStore", "Transition"]
