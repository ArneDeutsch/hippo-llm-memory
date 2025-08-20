"""PlaceGraph extensions providing subgraph queries.

Summary
-------
Adds a helper around :class:`hippo_mem.spatial.map.PlaceGraph` to fetch
local neighborhoods for retrieval. Nodes are returned in breadth-first
order starting from a reference context; edges follow the order in which
they are discovered. The traversal is deterministic by sorting neighbor
identifiers.
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

from .map import PlaceGraph as _PlaceGraph


class PlaceGraph(_PlaceGraph):
    """PlaceGraph with a ``local`` query for radius-based subgraphs."""

    def local(self, context: str, radius: int = 1) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Return node ids and directed edges within ``radius`` hops.

        Parameters
        ----------
        context:
            Reference context name.
        radius:
            Hop distance, by default ``1``.

        Returns
        -------
        tuple[list[int], list[tuple[int, int]]]
            Node identifiers and ``(src, dst)`` edge pairs.
        """

        if context not in self._context_to_id:
            return [], []
        start = self._context_to_id[context]
        nodes = [start]
        edges: List[Tuple[int, int]] = []
        visited = {start}
        q: deque[Tuple[int, int]] = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            if dist == radius:
                continue
            for nbr in sorted(self.graph.get(node, {})):
                edges.append((node, nbr))
                if nbr not in visited:
                    visited.add(nbr)
                    nodes.append(nbr)
                    q.append((nbr, dist + 1))
        return nodes, edges


__all__ = ["PlaceGraph"]
