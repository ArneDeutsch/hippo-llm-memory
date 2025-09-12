# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import heapq
from typing import Dict, List, Optional

from hippo_mem.spatial.map import PlaceGraph


class PathPlanner:
    """A* planner operating on a :class:`PlaceGraph`."""

    def __init__(self, graph: PlaceGraph) -> None:
        self.graph = graph

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """Return minimal-cost path from ``start`` to ``goal``.

        Parameters
        ----------
        start:
            Source context name.
        goal:
            Destination context name.

        Returns
        -------
        list[str]
            Sequence of context names including ``start`` and ``goal``.
        """

        if start not in self.graph._context_to_id or goal not in self.graph._context_to_id:
            return []
        start_id = self.graph._context_to_id[start]
        goal_id = self.graph._context_to_id[goal]

        def heuristic(a: int, b: int) -> float:
            _ax, _ay = self.graph.encoder.encode(self.graph._id_to_context[a]).coord
            _bx, _by = self.graph.encoder.encode(self.graph._id_to_context[b]).coord
            return 0.0

        open_set: list[tuple[float, float, int, Optional[int]]] = []
        heapq.heappush(open_set, (heuristic(start_id, goal_id), 0.0, start_id, None))
        came_from: Dict[int, Optional[int]] = {}
        costs = {start_id: 0.0}
        while open_set:
            _, g, node, parent = heapq.heappop(open_set)
            if node in came_from:
                continue
            came_from[node] = parent
            if node == goal_id:
                break
            for nbr, edge in self.graph.graph.get(node, {}).items():
                ng = g + edge.cost
                if ng < costs.get(nbr, float("inf")):
                    costs[nbr] = ng
                    heapq.heappush(open_set, (ng + heuristic(nbr, goal_id), ng, nbr, node))
        if goal_id not in came_from:
            return []
        path_ids: List[int] = []
        node = goal_id
        while node is not None:
            path_ids.append(node)
            node = came_from[node]
        path_ids.reverse()
        return [self.graph._id_to_context[i] for i in path_ids]


__all__ = ["PathPlanner"]
