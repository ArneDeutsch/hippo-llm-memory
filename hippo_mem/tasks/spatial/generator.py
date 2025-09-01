from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Set, Tuple

from .prompt import make_prompt

Move = str

_MOVES: list[Tuple[int, int, Move]] = [
    (-1, 0, "L"),
    (1, 0, "R"),
    (0, -1, "U"),
    (0, 1, "D"),
]


def _bfs(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid_size: int,
    obstacles: Set[Tuple[int, int]],
) -> List[Move] | None:
    """Return shortest path as moves using BFS."""

    queue: deque[Tuple[Tuple[int, int], List[Move]]] = deque([(start, [])])
    seen = {start}
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        for dx, dy, step in _MOVES:
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if (
                0 <= nx < grid_size
                and 0 <= ny < grid_size
                and nxt not in obstacles
                and nxt not in seen
            ):
                queue.append((nxt, path + [step]))
                seen.add(nxt)
    return None


def generate_spatial(
    size: int,
    seed: int,
    grid_size: int | None = None,
    obstacle_density: float | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Generate grid-world tasks with canonical move string answers."""

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

    def random_cell() -> Tuple[int, int]:
        return rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1)

    macros: List[Dict[str, object]] = []
    for _ in range(3):
        while True:
            start, goal = random_cell(), random_cell()
            if start == goal or start in obstacles or goal in obstacles:
                continue
            path = _bfs(start, goal, grid_size, obstacles)
            if path:
                macros.append({"start": start, "goal": goal, "steps": "".join(path)})
                break

    tasks: List[Dict[str, object]] = []
    for i in range(size):
        if i % 2 == 0:  # shortest path queries
            while True:
                start, goal = random_cell(), random_cell()
                if start == goal or start in obstacles or goal in obstacles:
                    continue
                path = _bfs(start, goal, grid_size, obstacles)
                if path:
                    prompt = make_prompt(grid_size, obstacles, start, goal)
                    tasks.append({"prompt": prompt, "answer": "".join(path)})
                    break
        else:  # macro path sequences
            macro = rng.choice(macros)
            prompt = make_prompt(grid_size, obstacles, macro["start"], macro["goal"])
            tasks.append({"prompt": prompt, "answer": macro["steps"]})

    return tasks
