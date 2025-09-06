import argparse
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict

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


SUITE_STRATEGIES: Dict[str, Callable[[argparse.Namespace, Dict[str, Any]], Dict[str, Any]]] = {
    "semantic_mem": lambda a, c: {
        "hop_depth": c.get("hop_depth", a.hop_depth),
        "inject_contradictions": c.get("inject_contradictions", a.contradict),
        "distractors": c.get("distractors", a.distractors),
        "entity_pool": c.get("entity_pool", a.entity_pool),
        "paraphrase_prob": c.get("paraphrase_prob", a.paraphrase_prob),
        "ambiguity_prob": c.get("ambiguity_prob", a.ambiguity_prob),
    },
    "episodic_cross_mem": lambda a, c: {
        "entity_pool": c.get("entity_pool", a.entity_pool),
        "distractors": c.get("distractors", a.distractors or None),
    },
    "spatial_multi": lambda a, c: {
        "grid_size": c.get("grid_size", a.grid_size),
        "obstacle_density": c.get("obstacle_density", a.obstacle_density),
    },
}


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit early if ``size`` or ``seed`` are invalid."""
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")


def main() -> None:
    """CLI entry point for building memory-first synthetic datasets."""
    parser = argparse.ArgumentParser(description="Build synthetic evaluation data")
    parser.add_argument("--suite", choices=list(SUITE_TO_GENERATOR.keys()), required=True)
    parser.add_argument("--size", type=int, default=100, help="Number of teach/test items")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--profile",
        choices=["easy", "default", "hard"],
        default="default",
        help="Difficulty profile to apply",
    )
    parser.add_argument("--distractors", type=int, default=0, help="Distractor sentences")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size for spatial_multi")
    parser.add_argument(
        "--obstacle-density", type=float, default=0.2, help="Obstacle density for spatial_multi"
    )
    parser.add_argument("--hop-depth", type=int, default=2, help="Hop depth for semantic_mem")
    parser.add_argument(
        "--contradict", action="store_true", help="Inject contradictions for semantic_mem"
    )
    parser.add_argument(
        "--entity-pool",
        type=int,
        default=4,
        help="Entity pool size for semantic_mem or episodic_cross_mem",
    )
    parser.add_argument(
        "--paraphrase-prob", type=float, default=0.0, help="Paraphrase probability for semantic_mem"
    )
    parser.add_argument(
        "--ambiguity-prob",
        type=float,
        default=0.0,
        help="Pronoun ambiguity probability for semantic_mem",
    )
    args = parser.parse_args()
    _validate_args(parser, args)

    profile_cfg = _load_profile(args.profile).get(args.suite, {})
    build = SUITE_STRATEGIES.get(args.suite, lambda *_: {})
    kwargs = build(args, profile_cfg)
    generator = SUITE_TO_GENERATOR[args.suite]
    items = generator(args.size, args.seed, profile=args.profile, **kwargs)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    teach_path = out_dir / f"{args.suite}_teach.jsonl"
    test_path = out_dir / f"{args.suite}_test.jsonl"
    write_jsonl(teach_path, items["teach"])
    write_jsonl(test_path, items["test"])
    checksum_path = out_dir / "checksums.json"
    version = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    d_teach = record_checksum(teach_path, checksum_path)
    update_dataset_card(args.suite, out_dir, teach_path.name, d_teach, version)
    d_test = record_checksum(test_path, checksum_path)
    update_dataset_card(args.suite, out_dir, test_path.name, d_test, version)

    if args.suite != "spatial_multi":
        keywords = {t["fact"] for t in items["teach"]}
        for entry in items["test"]:
            text = entry.get("prompt", "")
            if any(k in text for k in keywords):
                raise SystemExit(f"Leak detected in test prompt: {text}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
