"""Simple topological map with basic planning utilities.

This module provides :class:`PlaceGraph`, a very small graph structure
for experiments with spatial memory.  Nodes are created from textual
contexts via :meth:`observe` and assigned integer identifiers.  Each
edge stores a transition ``cost`` and a ``success`` probability.  A
tiny planner using either A* (default) or Dijkstra's algorithm is
included to compute shortest paths between contexts.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Edge:
    """Connection information between two places."""

    cost: float = 1.0
    success: float = 1.0
    last_seen: int = 0


@dataclass
class Place:
    """A place in the environment with pseudo coordinates."""

    name: str
    coord: Tuple[float, float]
    last_seen: int = 0


class ContextEncoder:
    """Deterministic context→coordinate encoder.

    The encoder maps unique context strings to pseudo coordinates
    derived from the hash of the string.  This keeps positions stable
    across runs without relying on any external resources.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Place] = {}

    def encode(self, context: str) -> Place:
        if context not in self._cache:
            h = hash(context)
            x = (h & 0xFFFF) / 1000.0
            y = ((h >> 16) & 0xFFFF) / 1000.0
            self._cache[context] = Place(context, (x, y))
        return self._cache[context]


class PlaceGraph:
    """Graph of observed places with light‑weight planning."""

    def __init__(self, path_integration: bool = False) -> None:
        self.encoder = ContextEncoder()
        self._context_to_id: Dict[str, int] = {}
        self._id_to_context: Dict[int, str] = {}
        # adjacency list: node -> neighbour -> Edge
        self.graph: Dict[int, Dict[int, Edge]] = {}
        self._next_id = 0
        self._last_obs: Optional[int] = None
        self._step = 0
        self._path_integration = path_integration
        self._position = (0.0, 0.0)
        self._last_coord: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Observation and graph construction
    def _ensure_node(self, context: str) -> int:
        node = self._context_to_id.get(context)
        if node is None:
            node = self._next_id
            self._next_id += 1
            self._context_to_id[context] = node
            self._id_to_context[node] = context
            self.graph.setdefault(node, {})
            # Encode for side effects (cache coordinates)
            self.encoder.encode(context)
        return node

    def observe(self, context: str) -> int:
        """Insert *context* into the graph and connect from previous.

        Observing a sequence of contexts grows the graph deterministically
        and adds an undirected edge between consecutive observations with
        unit cost and success probability of one.
        """

        self._step += 1
        node = self._ensure_node(context)

        place = self.encoder.encode(context)
        if self._path_integration:
            coord = place.coord
            if self._last_coord is None:
                self._position = (0.0, 0.0)
                self._last_coord = coord
            else:
                dx = coord[0] - self._last_coord[0]
                dy = coord[1] - self._last_coord[1]
                self._position = (self._position[0] + dx, self._position[1] + dy)
                self._last_coord = coord
            place.coord = self._position
        place.last_seen = self._step

        if self._last_obs is not None and self._last_obs != node:
            self._add_edge(self._last_obs, node, step=self._step)
            self._add_edge(node, self._last_obs, step=self._step)
        self._last_obs = node
        return node

    def connect(
        self,
        a_context: str,
        b_context: str,
        cost: float = 1.0,
        success: float = 1.0,
    ) -> None:
        """Explicitly connect two contexts."""

        a = self._ensure_node(a_context)
        b = self._ensure_node(b_context)
        self._add_edge(a, b, cost, success)
        self._add_edge(b, a, cost, success)

    def _add_edge(
        self,
        a: int,
        b: int,
        cost: float = 1.0,
        success: float = 1.0,
        *,
        step: Optional[int] = None,
    ) -> None:
        edge = self.graph[a].get(b)
        if edge is None:
            self.graph[a][b] = Edge(cost, success, last_seen=step or self._step)
        else:
            edge.cost = cost
            edge.success = success
            edge.last_seen = step or self._step

    # ------------------------------------------------------------------
    # Planning utilities
    def plan(self, start: str, goal: str, method: str = "astar") -> List[str]:
        """Return the shortest path between *start* and *goal* contexts."""

        s = self._ensure_node(start)
        g = self._ensure_node(goal)
        if method == "astar":
            path_ids = self._a_star(s, g)
        elif method == "dijkstra":
            path_ids = self._dijkstra(s, g)
        else:
            raise ValueError(f"Unknown planning method: {method}")
        return [self._id_to_context[i] for i in path_ids]

    # -- heuristics -----------------------------------------------------
    def _heuristic(self, a: int, b: int) -> float:
        pa = self.encoder.encode(self._id_to_context[a]).coord
        pb = self.encoder.encode(self._id_to_context[b]).coord
        return math.dist(pa, pb)

    def _reconstruct(self, came_from: Dict[int, int], current: int) -> List[int]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _a_star(self, start: int, goal: int) -> List[int]:
        open_set: List[Tuple[float, int]] = [(0.0, start)]
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {start: 0.0}
        f_score: Dict[int, float] = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)
            for neighbor, edge in self.graph.get(current, {}).items():
                tentative = g_score[current] + edge.cost
                if tentative < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def _dijkstra(self, start: int, goal: int) -> List[int]:
        queue: List[Tuple[float, int]] = [(0.0, start)]
        came_from: Dict[int, int] = {}
        costs: Dict[int, float] = {start: 0.0}
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                return self._reconstruct(came_from, node)
            if cost > costs.get(node, math.inf):
                continue
            for neighbor, edge in self.graph.get(node, {}).items():
                new_cost = cost + edge.cost
                if new_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    came_from[neighbor] = node
                    heapq.heappush(queue, (new_cost, neighbor))
        return []
