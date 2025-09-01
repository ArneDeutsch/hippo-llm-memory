from __future__ import annotations

from typing import Iterable, Tuple


def make_prompt(
    grid_size: int,
    obstacles: Iterable[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> str:
    """Return a prompt instructing for canonical move string output."""

    obs_list = sorted(obstacles)
    return (
        f"Grid {grid_size}x{grid_size} with obstacles {obs_list}. "
        f"Start {start} goal {goal}. "
        "Respond with moves using U, D, L, R only, no spaces."
    )
