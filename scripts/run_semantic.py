# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""CLI for running the semantic suite with KG dependency and ablation."""

from __future__ import annotations

import argparse

from hippo_eval.datasets import generate_semantic
from hippo_mem.memory import evaluate_semantic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic suite with optional KG ablation")
    parser.add_argument("--size", type=int, default=20, help="Number of items to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for dataset generation")
    parser.add_argument("--ablate", action="store_true", help="Evaluate with an empty KG")
    args = parser.parse_args()

    data = generate_semantic(args.size, args.seed, require_memory=True)
    em = evaluate_semantic(data, use_kg=not args.ablate)
    print(f"em={em:.3f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
