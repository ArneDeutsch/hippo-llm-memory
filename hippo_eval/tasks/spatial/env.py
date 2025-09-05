from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Set, Tuple

Move = str
_MOVES: list[tuple[int, int, Move]] = [
    (-1, 0, "L"),
    (1, 0, "R"),
    (0, -1, "U"),
    (0, 1, "D"),
]


@dataclass
class GridEnv:
    """Simple grid world with obstacles and an optimal path oracle."""

    size: int
    obstacles: Set[Tuple[int, int]]

    def shortest_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Move] | None:
        """Return shortest path as a list of moves using BFS."""

        queue: list[tuple[Tuple[int, int], List[Move]]] = [(start, [])]
        seen = {start}
        while queue:
            (x, y), path = queue.pop(0)
            if (x, y) == goal:
                return path
            for dx, dy, step in _MOVES:
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and nxt not in self.obstacles
                    and nxt not in seen
                ):
                    queue.append((nxt, path + [step]))
                    seen.add(nxt)
        return None

    @classmethod
    def sample(
        cls,
        seed: int,
        *,
        grid_size: int | None = None,
        obstacle_density: float | None = None,
        profile: str = "default",
    ) -> "GridEnv":
        """Return a random :class:`GridEnv` sampled from ``seed``."""

        rng = random.Random(seed)
        if grid_size is None:
            grid_size = {"easy": 5, "default": 6, "hard": 7}[profile]
        if obstacle_density is None:
            obstacle_density = {"easy": 0.15, "default": 0.25, "hard": 0.3}[profile]
        obstacles: Set[Tuple[int, int]] = set()
        for x in range(grid_size):
            for y in range(grid_size):
                if rng.random() < obstacle_density:
                    obstacles.add((x, y))
        return cls(size=grid_size, obstacles=obstacles)
