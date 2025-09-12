# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import random
from typing import Dict, List

from .env import GridEnv
from .prompt import make_prompt


def _sample_episode(
    env: GridEnv, rng: random.Random, episode_idx: int, topology_id: str
) -> Dict[str, object]:
    """Return a single episode task for ``env``."""

    while True:
        start = rng.randint(0, env.size - 1), rng.randint(0, env.size - 1)
        goal = rng.randint(0, env.size - 1), rng.randint(0, env.size - 1)
        if start == goal or start in env.obstacles or goal in env.obstacles:
            continue
        path = env.shortest_path(start, goal)
        if path:
            prompt = make_prompt(env.size, env.obstacles, start, goal)
            return {
                "prompt": prompt,
                "answer": "".join(path),
                "episode_id": f"{episode_idx:03d}",
                "context_key": topology_id,
            }


def generate_spatial_multi(
    num_teach: int,
    num_test: int,
    *,
    seed: int,
    grid_size: int | None = None,
    obstacle_density: float | None = None,
    profile: str = "default",
) -> Dict[str, List[Dict[str, object]]]:
    """Generate multi-episode spatial tasks for teach/test splits."""

    rng = random.Random(seed)
    env = GridEnv.sample(
        seed,
        grid_size=grid_size,
        obstacle_density=obstacle_density,
        profile=profile,
    )
    topology_id = f"grid-{seed}"
    teach = [_sample_episode(env, rng, i, topology_id) for i in range(num_teach)]
    test = [_sample_episode(env, rng, i + num_teach, topology_id) for i in range(num_test)]
    return {"topology_id": topology_id, "teach": teach, "test": test}
