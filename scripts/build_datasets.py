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
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


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


def generate_semantic(size: int, seed: int) -> List[Dict[str, str]]:
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
    for _ in range(size):
        who = rng.choice(people)
        item = rng.choice(items)
        store = rng.choice(stores)
        city = rng.choice(cities)

        text = f"{who} bought a {item} at {store}. {store} is in {city}."
        question = f"In which city did {who} buy the {item}?"
        tasks.append({"prompt": f"{text} {question}", "answer": city})

    return tasks


def generate_spatial(size: int, seed: int) -> List[Dict[str, int]]:
    """Generate grid based shortest-path questions.

    Each task asks for the Manhattan distance between two coordinates in a
    5×5 grid.
    """

    rng = random.Random(seed)
    grid = 5
    tasks: List[Dict[str, int]] = []
    for _ in range(size):
        x1, y1 = rng.randint(0, grid - 1), rng.randint(0, grid - 1)
        x2, y2 = rng.randint(0, grid - 1), rng.randint(0, grid - 1)
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


def generate_dataset(suite: str, size: int, seed: int) -> List[Dict[str, object]]:
    """Dispatch to the generator for ``suite``.

    This helper simplifies programmatic use and is exercised in unit tests to
    ensure all suites are deterministic for a given ``seed``.
    """

    try:
        generator = SUITE_TO_GENERATOR[suite]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown suite: {suite}") from exc
    return generator(size, seed)


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
    """

    parser = argparse.ArgumentParser(description="Build synthetic evaluation data")
    parser.add_argument("--suite", choices=SUITE_TO_GENERATOR.keys(), required=True)
    parser.add_argument("--size", type=int, default=100, help="Number of items")
    parser.add_argument("--n", dest="size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args()

    generator = SUITE_TO_GENERATOR[args.suite]
    items = generator(args.size, args.seed)
    write_jsonl(args.out, items)
    checksum_path = args.out.parent / "checksums.txt"
    record_checksum(args.out, checksum_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
