"""Macro planning utilities for spatial memory.

The real project would learn reusable action sequences ("macros") from
demonstrations.  Here we implement only the minimum scaffolding needed
for testing: a tiny in-memory library and a behaviour-cloning stub that
stores the shortest demonstration trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .map import SpatialMap


def plan_route(world: SpatialMap, start: str, end: str) -> List[str]:
    """Plan a route between two contexts using Dijkstra's algorithm."""

    return world.dijkstra(start, end)


@dataclass
class Macro:
    """Representation of a macro behaviour."""

    name: str
    trajectory: List[str]


class MacroLibrary:
    """Collection of macros with a behaviour cloning placeholder."""

    def __init__(self) -> None:
        self._macros: Dict[str, Macro] = {}

    def add(self, macro: Macro) -> None:
        self._macros[macro.name] = macro

    def get(self, name: str) -> Macro:
        return self._macros[name]

    def behavior_clone(self, name: str, demos: Sequence[List[str]]) -> Macro:
        """Return a macro cloned from demonstration trajectories.

        The simplest possible approach is used here: the shortest
        demonstration is taken as-is and stored as a macro.
        """

        if not demos:
            raise ValueError("No demonstrations provided")
        trajectory = list(min(demos, key=len))
        macro = Macro(name, trajectory)
        self.add(macro)
        return macro
