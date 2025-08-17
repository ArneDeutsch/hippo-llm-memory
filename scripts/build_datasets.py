"""Synthetic task generators used by the evaluation harness.

The real project builds fairly involved generators for episodic, semantic and
spatial suites.  For unit tests and CI we only require lightweight synthetic
data which still exercises the metric plumbing.  The functions below implement
deterministic generators for the three suites described in ``EVAL_PLAN.md``.

They can be used programmatically or via the ``main`` CLI entry point which
writes the generated items to a JSONL file.  The semantic generator supports
multi-hop chains and optional contradiction injection which can be toggled via
CLI flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set


def generate_episodic(size: int, seed: int) -> List[Dict[str, str]]:
    """Generate ``size`` W4 stories with associated queries.

    Each item is a dictionary with ``prompt`` and ``answer`` fields.  The
    generator is completely deterministic given ``seed``.
    """

    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"]
    actions = ["met", "saw", "helped", "found"]
    places = ["Cafe", "Library", "Park", "Mall"]
    times = ["Monday", "Tuesday", "Wednesday", "Thursday"]
    qtypes = ["who_at_where", "what_did_who", "where_was_who", "when_was_who"]

    tasks: List[Dict[str, str]] = []
    for _ in range(size):
        who = rng.choice(people)
        what = rng.choice(actions)
        where = rng.choice(places)
        when = rng.choice(times)

        story = f"{who} {what} at the {where} on {when}."
        qtype = rng.choice(qtypes)

        if qtype == "who_at_where":
            question = f"Who was at the {where}?"
            answer = who
        elif qtype == "what_did_who":
            question = f"What did {who} do?"
            answer = what
        elif qtype == "where_was_who":
            question = f"Where was {who}?"
            answer = f"the {where}"
        else:  # when_was_who
            question = f"When was {who} at the {where}?"
            answer = when

        tasks.append({"prompt": f"{story} {question}", "answer": answer})

    return tasks


def generate_semantic(
    size: int, seed: int, hops: int = 2, contradictions: bool = False
) -> List[Dict[str, str]]:
    """Generate 2–3 hop fact chains with optional contradictions.

    ``hops`` controls the chain depth linking people, items, stores and
    cities.  When ``hops`` is ``3`` an intermediate fact connecting the item
    to the store is added.  If ``contradictions`` is ``True`` a second,
    conflicting store→city fact is appended which the query requires
    disambiguating.
    """

    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"]
    items = ["book", "apple", "ball", "coin"]
    stores = ["StoreA", "StoreB", "StoreC"]
    cities = ["Paris", "London", "Rome", "Berlin"]

    tasks: List[Dict[str, str]] = []
    for _ in range(size):
        who = rng.choice(people)
        item = rng.choice(items)
        store = rng.choice(stores)
        city = rng.choice(cities)

        sentences: List[str] = []
        if hops == 2:
            sentences.append(f"{who} bought a {item} at {store}.")
        else:  # hops == 3
            sentences.append(f"{who} bought a {item}.")
            sentences.append(f"The {item} was sold at {store}.")
        sentences.append(f"{store} is in {city}.")

        if contradictions:
            false_city = rng.choice([c for c in cities if c != city])
            sentences.append(f"However, others report {store} is in {false_city}.")

        text = " ".join(sentences)
        if hops == 2:
            question = f"In which city did {who} buy the {item}?"
        else:
            question = f"In which city was the {item} that {who} bought sold?"

        tasks.append({"prompt": f"{text} {question}", "answer": city})

    return tasks


def generate_spatial(
    size: int,
    seed: int,
    grid_size: int = 5,
    obstacle_density: float = 0.2,
) -> List[Dict[str, object]]:
    """Generate grid-world tasks with obstacles and macro paths.

    Half of the tasks ask for the shortest path length between two coordinates
    while avoiding obstacles.  The other half repeat **macro paths**—pre-computed
    shortest routes between fixed start/goal pairs—to encourage procedural
    sequence learning.
    """

    from collections import deque

    rng = random.Random(seed)

    obstacles: Set[tuple[int, int]] = set()
    for x in range(grid_size):
        for y in range(grid_size):
            if rng.random() < obstacle_density:
                obstacles.add((x, y))

    def bfs(start: tuple[int, int], goal: tuple[int, int]) -> List[str] | None:
        queue: deque[tuple[tuple[int, int], List[str]]] = deque([(start, [])])
        seen = {start}
        moves = [(-1, 0, "L"), (1, 0, "R"), (0, -1, "U"), (0, 1, "D")]
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path
            for dx, dy, step in moves:
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

    def random_cell() -> tuple[int, int]:
        return rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1)

    macros: List[Dict[str, object]] = []
    for _ in range(3):
        while True:
            start, goal = random_cell(), random_cell()
            if start == goal or start in obstacles or goal in obstacles:
                continue
            path = bfs(start, goal)
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
                path = bfs(start, goal)
                if path:
                    prompt = (
                        f"Grid {grid_size}x{grid_size} with obstacles {sorted(obstacles)}. "
                        f"Start {start} goal {goal}. What is the shortest path length?"
                    )
                    tasks.append({"prompt": prompt, "answer": len(path)})
                    break
        else:  # macro path sequences
            macro = rng.choice(macros)
            prompt = (
                f"Grid {grid_size}x{grid_size} with obstacles {sorted(obstacles)}. "
                f"What move sequence leads from {macro['start']} to {macro['goal']}?"
            )
            tasks.append({"prompt": prompt, "answer": macro["steps"]})

    return tasks


SUITE_TO_GENERATOR = {
    "episodic": generate_episodic,
    "semantic": generate_semantic,
    "spatial": generate_spatial,
}


def generate_dataset(suite: str, size: int, seed: int, **kwargs: object) -> List[Dict[str, object]]:
    """Dispatch to the generator for ``suite``.

    Extra keyword arguments are forwarded to the underlying generator which
    allows callers to tweak suite-specific options such as ``hops`` or
    ``contradictions`` for the semantic tasks or grid size or obstacle
    density for the spatial suite.
    """

    try:
        generator = SUITE_TO_GENERATOR[suite]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown suite: {suite}") from exc
    return generator(size, seed, **kwargs)


def write_jsonl(path: Path, items: Iterable[Dict[str, object]]) -> None:
    """Write items to ``path`` in JSON Lines format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj) + "\n")


def sha256_file(path: Path) -> str:
    """Return the SHA256 checksum of ``path``."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def record_checksum(data_path: Path, checksum_file: Path) -> str:
    """Append SHA256 of ``data_path`` to ``checksum_file`` and return it."""

    digest = sha256_file(data_path)
    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    with checksum_file.open("a", encoding="utf-8") as f:
        f.write(f"{digest}  {data_path.name}\n")
    return digest


def main() -> None:
    """CLI entry point for building small synthetic datasets.

    Example:

    ``python scripts/build_datasets.py --suite episodic --size 100 --seed 42 \
    --out data/episodic_100_42.jsonl``

    For the spatial suite additional parameters control the grid world:

    ``python scripts/build_datasets.py --suite spatial --size 50 --seed 0 \
    --grid-size 7 --obstacle-density 0.3 --out data/spatial.jsonl``
    """

    parser = argparse.ArgumentParser(description="Build synthetic evaluation data")
    parser.add_argument("--suite", choices=SUITE_TO_GENERATOR.keys(), required=True)
    parser.add_argument("--size", type=int, default=100, help="Number of items")
    parser.add_argument("--n", dest="size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--hops", type=int, choices=[2, 3], default=2, help="Hop depth for semantic facts"
    )
    parser.add_argument(
        "--contradictions",
        action="store_true",
        help="Inject contradictory statements in semantic suite",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Grid size for spatial suite",
    )
    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=0.2,
        help="Obstacle density for spatial suite",
    )
    args = parser.parse_args()

    generator = SUITE_TO_GENERATOR[args.suite]
    if args.suite == "semantic":
        items = generator(
            args.size,
            args.seed,
            hops=args.hops,
            contradictions=args.contradictions,
    elif args.suite == "spatial":
        items = generator(
            args.size,
            args.seed,
            grid_size=args.grid_size,
            obstacle_density=args.obstacle_density,
        )
    else:
        items = generator(args.size, args.seed)
    write_jsonl(args.out, items)
    checksum_path = args.out.parent / "checksums.txt"
    record_checksum(args.out, checksum_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
