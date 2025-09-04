"""Command-line interface for dataset generation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from . import (
    SUITE_TO_GENERATOR,
    record_checksum,
    update_dataset_card,
    write_jsonl,
)


def _load_profile(name: str) -> Dict[str, Any]:
    """Return profile configuration for ``name`` or an empty dict."""

    cfg_dir = Path(__file__).resolve().parents[2] / "configs" / "datasets"
    path = cfg_dir / f"{name}.yaml"
    if not path.exists():  # pragma: no cover - defensive
        return {}
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def main() -> None:
    """CLI entry point for building small synthetic datasets."""

    parser = argparse.ArgumentParser(description="Build synthetic evaluation datasets")
    parser.add_argument("--suite", required=True, choices=list(SUITE_TO_GENERATOR))
    parser.add_argument("--size", type=int, required=True, help="Number of items to generate")
    parser.add_argument("--seed", type=int, required=True, help="RNG seed")
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
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size for spatial suite")
    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=0.2,
        help="Obstacle density for spatial suite",
    )
    parser.add_argument(
        "--hop-depth", type=int, default=2, help="Hop depth (2 or 3) for semantic suite"
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
        max_corr = 2 if profile_cfg.get("corrections", True) else 0
        items = generator(
            args.size,
            args.seed,
            distractors=profile_cfg.get("distractors", args.distractors),
            max_corrections=max_corr,
            **common,
        )
    elif args.suite == "episodic_cross":
        items = generator(
            args.size,
            args.seed,
            entity_pool=profile_cfg.get("entity_pool", args.entity_pool),
            distractors=profile_cfg.get("distractors", args.distractors or None),
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
            distractors=profile_cfg.get("distractors", args.distractors),
            entity_pool=profile_cfg.get("entity_pool", args.entity_pool),
            paraphrase_prob=profile_cfg.get("paraphrase_prob", args.paraphrase_prob),
            ambiguity_prob=profile_cfg.get("ambiguity_prob", args.ambiguity_prob),
            **common,
        )
    else:
        items = generator(args.size, args.seed, **common)
    write_jsonl(args.out, items)
    checksum_path = args.out.parent / "checksums.json"
    digest = record_checksum(args.out, checksum_path)
    version = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    update_dataset_card(args.suite, args.out.parent, args.out.name, digest, version)


__all__ = ["main"]
