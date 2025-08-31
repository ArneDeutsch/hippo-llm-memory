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
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from omegaconf import OmegaConf

SIZES = [50, 200, 1000]
SEEDS = [1337, 2025, 4242]


def generate_episodic(
    size: int,
    seed: int,
    distractors: int | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Generate ``size`` W4 stories with optional distractors and swaps.

    Parameters
    ----------
    size:
        Number of items to generate.
    seed:
        RNG seed.
    distractors:
        Optional count of distractor sentences to insert *before* the target
        fact. When ``None`` the count is derived from ``profile``.
    profile:
        Difficulty profile (``easy``/``default``/``hard``). ``hard`` adds an
        extra distractor after the target event that reuses the same location
        with a different protagonist, forcing models to avoid relying on the
        last sentence.
    """

    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"]
    actions = ["met", "saw", "helped", "found"]
    places = ["Cafe", "Library", "Park", "Mall"]
    times = ["Monday", "Tuesday", "Wednesday", "Thursday"]
    qtypes = ["who_at_where", "what_did_who", "where_was_who", "when_was_who"]

    if distractors is None:
        distractors = {"easy": 0, "default": 2, "hard": 4}[profile]
    post_distractors = 1 if profile == "hard" else 0

    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        pre_sents: List[str] = []
        for _ in range(distractors):
            dwho = rng.choice(people)
            dwhat = rng.choice(actions)
            dwhere = rng.choice(places)
            dwhen = rng.choice(times)
            pre_sents.append(f"{dwho} {dwhat} at the {dwhere} on {dwhen}.")

        who = rng.choice(people)
        what = rng.choice(actions)
        where = rng.choice(places)
        when = rng.choice(times)
        target_sent = f"{who} {what} at the {where} on {when}."

        post_sents: List[str] = []
        for _ in range(post_distractors):
            pdwho = rng.choice([p for p in people if p != who])
            pdwhat = rng.choice(actions)
            pdwhen = rng.choice(times)
            post_sents.append(f"{pdwho} {pdwhat} at the {where} on {pdwhen}.")

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

        story = " ".join(pre_sents + [target_sent] + post_sents)
        reward = rng.random() < 0.3
        pin = rng.random() < 0.1
        tasks.append(
            {
                "prompt": f"{story} {question}",
                "answer": answer,
                "reward": reward,
                "pin": pin,
            }
        )

    return tasks


def generate_episodic_multi(
    size: int,
    seed: int,
    distractors: int | None = None,
    corrections: bool | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Multi-turn episodes with distractors and optional corrections."""

    if distractors is None:
        distractors = {"easy": 8, "default": 10, "hard": 12}[profile]
    if corrections is None:
        corrections = True

    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"]
    colors = ["red", "blue", "green", "yellow"]
    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        who = rng.choice(people)
        first_color = rng.choice(colors)
        distractor_sents = [
            f"Distractor: {rng.choice(people)} likes {rng.choice(colors)}."
            for _ in range(distractors)
        ]
        sents = distractor_sents + [f"{who} likes {first_color}."]
        if corrections:
            new_color = rng.choice([c for c in colors if c != first_color])
            sents.append(f"Actually, {who} likes {new_color}.")
            answer = new_color
        else:
            answer = first_color
        question = f"What color does {who} like?"
        prompt = " ".join(sents + [question])
        tasks.append({"prompt": prompt, "answer": answer})
    return tasks


def generate_episodic_cross(
    size: int,
    seed: int,
    entity_pool: int | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Cross-episode recall after a flush marker.

    Parameters
    ----------
    size:
        Number of items to generate.
    seed:
        RNG seed for determinism.
    entity_pool:
        Number of unique people/places to sample from. Increasing this raises
        task difficulty by reducing memorisation of a small fixed set. When
        ``None`` the value is derived from ``profile``.
    profile:
        Difficulty profile controlling the default ``entity_pool`` size.
    """

    rng = random.Random(seed)
    if entity_pool is None:
        entity_pool = {"easy": 4, "default": 6, "hard": 8}[profile]
    base_people = [
        "Alice",
        "Bob",
        "Carol",
        "Dave",
        "Eve",
        "Frank",
        "Grace",
        "Heidi",
    ]
    base_places = [
        "Cafe",
        "Library",
        "Park",
        "Mall",
        "Office",
        "School",
        "Cinema",
        "Museum",
    ]
    people = base_people[:entity_pool]
    places = base_places[:entity_pool]
    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        who = rng.choice(people)
        where = rng.choice(places)
        fact = f"{who} went to the {where}."
        prompt = f"{fact} --- FLUSH --- Where did {who} go?"
        tasks.append({"prompt": prompt, "answer": f"the {where}"})
    return tasks


def generate_episodic_capacity(
    size: int,
    seed: int,
    context_budget: int | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Episodes exceeding the decoding context budget.

    ``context_budget`` defaults depend on ``profile``.
    """

    rng = random.Random(seed)
    if context_budget is None:
        context_budget = {"easy": 256, "default": 384, "hard": 512}[profile]
    people = ["Alice", "Bob", "Carol", "Dave"]
    places = ["Cafe", "Library", "Park", "Mall"]
    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        who = rng.choice(people)
        where = rng.choice(places)
        fact = f"{who} went to the {where}."
        filler = " ".join("filler" for _ in range(context_budget + 10))
        question = f"Where did {who} go?"
        prompt = f"{fact} {filler} {question}"
        tasks.append({"prompt": prompt, "answer": f"the {where}"})
    return tasks


def generate_semantic(
    size: int,
    seed: int,
    hop_depth: int | None = None,
    inject_contradictions: bool | None = None,
    require_memory: bool = False,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Generate 2–3 hop fact chains linking people, items, stores and cities.

    Parameters
    ----------
    size:
        Number of items to generate.
    seed:
        RNG seed for determinism.
    hop_depth:
        ``2`` creates a purchase event and links the store to a city.
        ``3`` adds an intermediate hop linking the item to the store before
        linking the store to the city.
    inject_contradictions:
        If ``True``, an additional contradictory statement about the store's
        city is inserted, requiring disambiguation in the query.
    require_memory:
        When ``True``, the prompt omits fact sentences, forcing models to
        retrieve them from memory.  Facts are still returned in ``facts`` for
        ingestion during teach mode.
    profile:
        Difficulty profile controlling default ``hop_depth`` and contradiction
        behaviour.

    Returns
    -------
    list[dict[str, object]]
        ``prompt``/``answer`` pairs plus ``facts`` metadata.  Each fact carries
        a ``schema_fit`` label and ``time`` index for consolidation studies.
    """

    if hop_depth is None:
        hop_depth = 2 if profile != "hard" else 3
    if inject_contradictions is None:
        inject_contradictions = profile in {"default", "hard"}
    if hop_depth not in {2, 3}:
        raise ValueError("hop_depth must be 2 or 3")

    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"]
    items = ["book", "apple", "ball", "coin"]
    stores = ["StoreA", "StoreB", "StoreC"]
    cities = ["Paris", "London", "Rome", "Berlin"]

    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        who = rng.choice(people)
        item = rng.choice(items)
        store = rng.choice(stores)
        city = rng.choice(cities)

        parts: List[str] = []
        facts: List[Dict[str, object]] = []
        if hop_depth == 2:
            sent = f"{who} bought a {item} at {store}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": True, "time": len(facts)})
        else:  # hop_depth == 3
            sent = f"{who} bought a {item}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": True, "time": len(facts)})
            sent = f"The {item} was sold at {store}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": True, "time": len(facts)})
        sent = f"{store} is in {city}."
        if not require_memory:
            parts.append(sent)
        facts.append({"text": sent, "schema_fit": True, "time": len(facts)})

        if inject_contradictions:
            false_city = rng.choice([c for c in cities if c != city])
            sent = f"However, others report {store} is in {false_city}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": False, "time": len(facts)})
            question = f"Despite conflicting reports, in which city did {who} buy the {item}?"
        else:
            question = f"In which city did {who} buy the {item}?"

        text = " ".join(parts).strip()
        prompt = f"{text} {question}" if text else question
        tasks.append({"prompt": prompt, "answer": city, "facts": facts})

    return tasks


def generate_spatial(
    size: int,
    seed: int,
    grid_size: int | None = None,
    obstacle_density: float | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Generate grid-world tasks with obstacles, macros and trajectories.

    Defaults for ``grid_size`` and ``obstacle_density`` depend on ``profile``.

    A third of the tasks ask for the shortest path length between two
    coordinates while avoiding obstacles.  Another third repeat **macro paths**—
    pre-computed shortest routes between fixed start/goal pairs—to encourage
    procedural sequence learning.  The remaining tasks present random walk
    trajectories and ask for the final coordinate to stress path integration.
    """

    from collections import deque

    rng = random.Random(seed)

    if grid_size is None:
        grid_size = {"easy": 5, "default": 6, "hard": 7}[profile]
    if obstacle_density is None:
        obstacle_density = {"easy": 0.15, "default": 0.25, "hard": 0.3}[profile]

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
        if i % 3 == 0:  # shortest path queries
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
        elif i % 3 == 1:  # macro path sequences
            macro = rng.choice(macros)
            prompt = (
                f"Grid {grid_size}x{grid_size} with obstacles {sorted(obstacles)}. "
                f"What move sequence leads from {macro['start']} to {macro['goal']}?"
            )
            tasks.append({"prompt": prompt, "answer": macro["steps"]})
        else:  # random walk trajectory
            while True:
                start = random_cell()
                if start in obstacles:
                    continue
                pos = start
                trajectory = [start]
                moves = []
                for _ in range(5):
                    options = []
                    for dx, dy, step in [(-1, 0, "L"), (1, 0, "R"), (0, -1, "U"), (0, 1, "D")]:
                        nx, ny = pos[0] + dx, pos[1] + dy
                        nxt = (nx, ny)
                        if 0 <= nx < grid_size and 0 <= ny < grid_size and nxt not in obstacles:
                            options.append((dx, dy, step))
                    if not options:
                        break
                    dx, dy, step = rng.choice(options)
                    pos = (pos[0] + dx, pos[1] + dy)
                    trajectory.append(pos)
                    moves.append(step)
                if moves:
                    prompt = (
                        f"Grid {grid_size}x{grid_size} with obstacles {sorted(obstacles)}. "
                        f"Start {start}. After moves {''.join(moves)} where do you end?"
                    )
                    tasks.append({"prompt": prompt, "answer": pos, "trajectory": trajectory})
                    break

    return tasks


SUITE_TO_GENERATOR = {
    "episodic": generate_episodic,
    "semantic": generate_semantic,
    "spatial": generate_spatial,
    "episodic_multi": generate_episodic_multi,
    "episodic_cross": generate_episodic_cross,
    "episodic_capacity": generate_episodic_capacity,
}


# kept: used by tests/test_datasets.py
def generate_dataset(
    suite: str, size: int, seed: int, profile: str = "default", **kwargs: object
) -> List[Dict[str, object]]:
    """Dispatch to the generator for ``suite`` with a difficulty profile.

    Additional keyword arguments are forwarded to the underlying generator,
    allowing callers to customise parameters such as grid size or obstacle
    density for the spatial suite.
    """

    try:
        generator = SUITE_TO_GENERATOR[suite]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown suite: {suite}") from exc
    return generator(size, seed, profile=profile, **kwargs)


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
    """Record SHA256 of ``data_path`` in ``checksum_file`` and return it."""

    digest = sha256_file(data_path)
    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, str] = {}
    if checksum_file.exists():
        data = json.loads(checksum_file.read_text())
    data[data_path.name] = digest
    checksum_file.write_text(json.dumps(data, indent=2))
    return digest


def update_dataset_card(
    suite: str,
    suite_dir: Path,
    filename: str,
    digest: str,
    generator_version: str,
) -> None:
    """Update ``dataset_card.json`` for ``suite`` with ``filename`` → ``digest``."""

    card_path = suite_dir / "dataset_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
    else:
        card = {
            "suite": suite,
            "sizes": SIZES,
            "seeds": SEEDS,
            "generator_version": generator_version,
            "files": {},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cli_example": (
                f"python scripts/make_datasets.py suite={suite} size=<size> seed=<seed> "
                f"out=data/{suite}/<size>_<seed>.jsonl"
            ),
        }
    card["files"][filename] = digest
    card_path.write_text(json.dumps(card, indent=2))


def _load_profile(name: str) -> Dict[str, Any]:
    """Return profile configuration for ``name`` or an empty dict."""

    cfg_dir = Path(__file__).resolve().parents[2] / "configs" / "datasets"
    path = cfg_dir / f"{name}.yaml"
    if not path.exists():  # pragma: no cover - defensive
        return {}
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def main() -> None:
    """CLI entry point for building small synthetic datasets.

    Example:

    ``python scripts/make_datasets.py --suite episodic --size 100 --seed 42 \
    --distractors 2 --out data/episodic_100_42.jsonl``

    For the spatial suite additional parameters control the grid world:

    ``python scripts/make_datasets.py --suite spatial --size 50 --seed 0 \
    --grid-size 7 --obstacle-density 0.3 --out data/spatial.jsonl``

    The semantic suite supports multi-hop chains and optional contradictions:

    ``python scripts/make_datasets.py --suite semantic --size 20 --seed 0 \
    --hop-depth 3 --contradict --out data/semantic.jsonl``
    """

    parser = argparse.ArgumentParser(description="Build synthetic evaluation data")
    parser.add_argument("--suite", choices=SUITE_TO_GENERATOR.keys(), required=True)
    parser.add_argument("--size", type=int, default=100, help="Number of items")
    parser.add_argument("--n", dest="size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--profile",
        choices=["easy", "default", "hard"],
        default="default",
        help="Difficulty profile to apply",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="Number of distractor sentences for episodic suite",
    )
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
    parser.add_argument(
        "--hop-depth",
        type=int,
        default=2,
        help="Hop depth (2 or 3) for semantic suite",
    )
    parser.add_argument(
        "--contradict",
        action="store_true",
        help="Inject contradictory store locations for semantic suite",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=256,
        help="Context budget for episodic_capacity suite",
    )
    parser.add_argument(
        "--entity-pool",
        type=int,
        default=4,
        help="Number of unique entities for episodic_cross suite",
    )
    args = parser.parse_args()
    profile_cfg = _load_profile(args.profile).get(args.suite, {})

    generator = SUITE_TO_GENERATOR[args.suite]
    common = {"profile": args.profile}
    if args.suite == "spatial":
        items = generator(
            args.size,
            args.seed,
            grid_size=profile_cfg.get("grid_size", args.grid_size),
            obstacle_density=profile_cfg.get("obstacle_density", args.obstacle_density),
            **common,
        )
    elif args.suite == "episodic":
        items = generator(
            args.size,
            args.seed,
            distractors=profile_cfg.get("distractors", args.distractors),
            **common,
        )
    elif args.suite == "episodic_multi":
        items = generator(
            args.size,
            args.seed,
            distractors=profile_cfg.get("distractors", args.distractors),
            corrections=profile_cfg.get("corrections", True),
            **common,
        )
    elif args.suite == "episodic_cross":
        items = generator(
            args.size,
            args.seed,
            entity_pool=profile_cfg.get("entity_pool", args.entity_pool),
            **common,
        )
    elif args.suite == "episodic_capacity":
        items = generator(
            args.size,
            args.seed,
            context_budget=profile_cfg.get("context_budget", args.context_budget),
            **common,
        )
    elif args.suite == "semantic":
        items = generator(
            args.size,
            args.seed,
            hop_depth=profile_cfg.get("hop_depth", args.hop_depth),
            inject_contradictions=profile_cfg.get("inject_contradictions", args.contradict),
            **common,
        )
    else:
        items = generator(args.size, args.seed, **common)
    write_jsonl(args.out, items)
    checksum_path = args.out.parent / "checksums.json"
    digest = record_checksum(args.out, checksum_path)
    version = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    update_dataset_card(args.suite, args.out.parent, args.out.name, digest, version)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
