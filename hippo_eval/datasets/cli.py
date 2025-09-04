"""CLI wrappers for synthetic dataset generation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict

from omegaconf import OmegaConf

from . import SUITE_TO_GENERATOR, record_checksum, update_dataset_card, write_jsonl


def _load_profile(name: str) -> Dict[str, Any]:
    """Return profile configuration for ``name`` or an empty dict."""
    cfg_dir = Path(__file__).resolve().parents[2] / "configs" / "datasets"
    path = cfg_dir / f"{name}.yaml"
    if not path.exists():  # pragma: no cover - defensive
        return {}
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


# Maps suite names to callables that build generator kwargs from CLI args and
# difficulty profiles. Extend this to support new suites.
SUITE_STRATEGIES: Dict[str, Callable[[argparse.Namespace, Dict[str, Any]], Dict[str, Any]]] = {
    "spatial": lambda a, c: {
        "grid_size": c.get("grid_size", a.grid_size),
        "obstacle_density": c.get("obstacle_density", a.obstacle_density),
    },
    "episodic": lambda a, c: {"distractors": c.get("distractors", a.distractors)},
    "episodic_multi": lambda a, c: {
        "distractors": c.get("distractors", a.distractors),
        "max_corrections": 2 if c.get("corrections", True) else 0,
    },
    "episodic_cross": lambda a, c: {
        "entity_pool": c.get("entity_pool", a.entity_pool),
        "distractors": c.get("distractors", a.distractors or None),
    },
    "episodic_capacity": lambda a, c: {"context_budget": c.get("context_budget", a.context_budget)},
    "semantic": lambda a, c: {
        "hop_depth": c.get("hop_depth", a.hop_depth),
        "inject_contradictions": c.get("inject_contradictions", a.contradict),
        "distractors": c.get("distractors", a.distractors),
        "entity_pool": c.get("entity_pool", a.entity_pool),
        "paraphrase_prob": c.get("paraphrase_prob", a.paraphrase_prob),
        "ambiguity_prob": c.get("ambiguity_prob", a.ambiguity_prob),
    },
}


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit early if ``size`` or ``seed`` are invalid."""
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")


def main() -> None:
    """CLI entry point for building small synthetic datasets."""
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
        help="Number of distractor sentences for episodic, episodic_cross or semantic suites",
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
        help="Number of unique entities for episodic_cross or semantic suites",
    )
    parser.add_argument(
        "--paraphrase-prob",
        type=float,
        default=0.0,
        help="Paraphrase probability for semantic suite",
    )
    parser.add_argument(
        "--ambiguity-prob",
        type=float,
        default=0.0,
        help="Pronoun ambiguity probability for semantic suite",
    )
    args = parser.parse_args()
    _validate_args(parser, args)

    profile_cfg = _load_profile(args.profile).get(args.suite, {})
    build = SUITE_STRATEGIES.get(args.suite, lambda *_: {})
    kwargs = build(args, profile_cfg)
    generator = SUITE_TO_GENERATOR[args.suite]
    items = generator(args.size, args.seed, profile=args.profile, **kwargs)
    write_jsonl(args.out, items)
    checksum_path = args.out.parent / "checksums.json"
    digest = record_checksum(args.out, checksum_path)
    version = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    update_dataset_card(args.suite, args.out.parent, args.out.name, digest, version)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
