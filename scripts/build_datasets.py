"""Synthetic task generators used by the evaluation harness.

The real project builds fairly involved generators for episodic, semantic and
spatial suites.  For unit tests and CI we only require lightweight synthetic
data which still exercises the metric plumbing.  The functions below implement
deterministic generators for the three suites described in ``EVAL_PLAN.md``.

They can be used programmatically or via the ``main`` CLI entry point which
writes the generated items to a JSONL file.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


def generate_episodic(n: int, seed: int) -> List[Dict[str, str]]:
    """Generate ``n`` W4 stories with associated queries.

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
    for _ in range(n):
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


def generate_semantic(n: int, seed: int) -> List[Dict[str, str]]:
    """Generate simple two-hop reasoning facts.

    The template creates a purchase event and links the store to a city.  The
    question requires resolving the city of purchase – a tiny multi-hop query.
    """

    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"]
    items = ["book", "apple", "ball", "coin"]
    stores = ["StoreA", "StoreB", "StoreC"]
    cities = ["Paris", "London", "Rome", "Berlin"]

    tasks: List[Dict[str, str]] = []
    for _ in range(n):
        who = rng.choice(people)
        item = rng.choice(items)
        store = rng.choice(stores)
        city = rng.choice(cities)

        text = f"{who} bought a {item} at {store}. {store} is in {city}."
        question = f"In which city did {who} buy the {item}?"
        tasks.append({"prompt": f"{text} {question}", "answer": city})

    return tasks


def generate_spatial(n: int, seed: int) -> List[Dict[str, int]]:
    """Generate grid based shortest-path questions.

    Each task asks for the Manhattan distance between two coordinates in a
    5×5 grid.
    """

    rng = random.Random(seed)
    size = 5
    tasks: List[Dict[str, int]] = []
    for _ in range(n):
        x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        x2, y2 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        prompt = (
            f"Start at ({x1},{y1}) and move to ({x2},{y2}). " "What is the shortest path length?"
        )
        answer = abs(x1 - x2) + abs(y1 - y2)
        tasks.append({"prompt": prompt, "answer": answer})

    return tasks


SUITE_TO_GENERATOR = {
    "episodic": generate_episodic,
    "semantic": generate_semantic,
    "spatial": generate_spatial,
}


def write_jsonl(path: Path, items: Iterable[Dict[str, object]]) -> None:
    """Write items to ``path`` in JSON Lines format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj) + "\n")


def main() -> None:
    """CLI entry point for building small synthetic datasets.

    Example:

    ``python scripts/build_datasets.py --suite episodic --n 100 --seed 42 \
    --out data/episodic.jsonl``
    """

    parser = argparse.ArgumentParser(description="Build synthetic evaluation data")
    parser.add_argument("--suite", choices=SUITE_TO_GENERATOR.keys(), required=True)
    parser.add_argument("--n", type=int, default=100, help="Number of items")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args()

    generator = SUITE_TO_GENERATOR[args.suite]
    items = generator(args.n, args.seed)
    write_jsonl(args.out, items)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
