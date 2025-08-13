"""Spatial map structures.

This module provides a tiny topological map that can encode textual
context strings into places and plan paths between them.  The
implementation is intentionally lightweight and serves primarily as a
stub for more sophisticated spatial memory systems.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Place:
    """A place in the environment."""

    name: str
    coord: Tuple[float, float]


class ContextEncoder:
    """Very small context→place encoder.

    The encoder simply maps unique context strings to deterministic
    pseudo-coordinates derived from the hash of the string.  It is
    sufficient for unit tests that require stable yet arbitrary
    positions.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Place] = {}

    def encode(self, context: str) -> Place:
        """Return a :class:`Place` for *context*."""

        if context not in self._cache:
            h = hash(context)
            # Derive pseudo coordinates from the hash to keep things
            # deterministic across calls.
            x = (h & 0xFFFF) / 1000.0
            y = ((h >> 16) & 0xFFFF) / 1000.0
            self._cache[context] = Place(context, (x, y))
        return self._cache[context]


class SpatialMap:
    """Topological map with basic path‑finding utilities."""

    def __init__(self) -> None:
        self.encoder = ContextEncoder()
        # adjacency list: node -> neighbor -> weight
        self.graph: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Places and edges
    def add_place(self, context: str) -> Place:
        """Register a context string as a place in the map."""

        place = self.encoder.encode(context)
        self.graph.setdefault(place.name, {})
        return place

    def connect(self, a_context: str, b_context: str, weight: Optional[float] = None) -> None:
        """Create an undirected edge between two contexts.

        Args:
            a_context: First context.
            b_context: Second context.
            weight: Optional explicit edge weight.  If not provided, the
                Euclidean distance between the encoded coordinates is
                used.
        """

        a = self.add_place(a_context)
        b = self.add_place(b_context)
        if weight is None:
            weight = math.dist(a.coord, b.coord)
        self.graph[a.name][b.name] = weight
        self.graph[b.name][a.name] = weight

    # ------------------------------------------------------------------
    # Path finding
    def shortest_path(self, start_context: str, end_context: str) -> List[str]:
        """Compute a path using the A* search algorithm."""

        self.add_place(start_context)
        self.add_place(end_context)
        return self._a_star(start_context, end_context)

    def _heuristic(self, a: str, b: str) -> float:
        pa = self.encoder.encode(a).coord
        pb = self.encoder.encode(b).coord
        return math.dist(pa, pb)

    def _reconstruct(self, came_from: Dict[str, str], current: str) -> List[str]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))

    def _a_star(self, start: str, goal: str) -> List[str]:
        open_set: List[Tuple[float, str]] = []
        heapq.heappush(open_set, (0.0, start))
        came_from: Dict[str, str] = {}
        g_score: Dict[str, float] = {start: 0.0}
        f_score: Dict[str, float] = {start: self._heuristic(start, goal)}

        visited: Set[str] = set()
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)
            visited.add(current)
            for neighbor, weight in self.graph.get(current, {}).items():
                tentative = g_score[current] + weight
                if tentative < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + self._heuristic(neighbor, goal)
                    if neighbor not in visited:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def dijkstra(self, start_context: str, end_context: str) -> List[str]:
        """Dijkstra's algorithm for comparison and testing."""

        self.add_place(start_context)
        self.add_place(end_context)
        start = start_context
        goal = end_context
        queue: List[Tuple[float, str]] = [(0.0, start)]
        came_from: Dict[str, str] = {}
        costs: Dict[str, float] = {start: 0.0}
        visited: Set[str] = set()
        while queue:
            cost, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if node == goal:
                return self._reconstruct(came_from, node)
            for neighbor, weight in self.graph.get(node, {}).items():
                new_cost = cost + weight
                if new_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    came_from[neighbor] = node
                    heapq.heappush(queue, (new_cost, neighbor))
        return []
