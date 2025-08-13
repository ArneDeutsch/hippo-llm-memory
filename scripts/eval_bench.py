"""Light‑weight evaluation harness used in tests and CI.

This module provides a tiny yet functional implementation of the evaluation
workflow described in :mod:`EVAL_PLAN.md`.  It does **not** attempt to run any
real language model – instead predictions are taken to be equal to the ground
truth answers.  The goal is to exercise the metric plumbing and file layout so
that higher level tooling can be validated without expensive model inference.

Example usage from the command line::

    python scripts/eval_bench.py --suite episodic --n 5 --seed 0 \
        --preset baselines/core

The resulting ``metrics.json``/``metrics.csv``/``meta.json`` files are written
to ``runs/<date>/<preset>/<suite>/`` by default or to the directory supplied via
``--outdir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from build_datasets import (
    SUITE_TO_GENERATOR,
)


def _git_sha() -> str:
    """Return the current git commit SHA, or ``unknown`` if unavailable."""

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:  # pragma: no cover - extremely unlikely in tests
        return "unknown"


def evaluate(suite: str, n: int, seed: int, preset: str, outdir: Path) -> None:
    """Run evaluation for ``suite`` and write metrics to ``outdir``."""

    generator = SUITE_TO_GENERATOR[suite]
    tasks = generator(n, seed)

    rows: List[Dict[str, object]] = []
    correct = 0
    total_tokens = 0
    for idx, item in enumerate(tasks):
        prompt = str(item["prompt"])
        answer = str(item["answer"])
        pred = str(item.get("pred", answer))  # echo model

        is_correct = int(pred.strip().lower() == answer.strip().lower())
        correct += is_correct
        total_tokens += len(prompt.split()) + len(answer.split())
        rows.append(
            {
                "idx": idx,
                "prompt": prompt,
                "answer": answer,
                "pred": pred,
                "correct": is_correct,
                "latency_ms": 0.0,
                "flags": "pre_replay",
            }
        )

    em = correct / n if n else 0.0
    metrics = {
        "suite": suite,
        "n": n,
        "seed": seed,
        "preset": preset,
        "metrics": {suite: {"em": em}, "compute": {"tokens": total_tokens}},
    }

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f)

    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "prompt",
                "answer",
                "pred",
                "correct",
                "latency_ms",
                "flags",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    meta = {"git_sha": _git_sha(), "model": "mock", "config_hash": "", "ablate": {}}
    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)


def main() -> None:
    """CLI entry point for the evaluation harness."""

    parser = argparse.ArgumentParser(description="Run synthetic evaluation")
    parser.add_argument("--suite", choices=SUITE_TO_GENERATOR.keys(), required=True)
    parser.add_argument("--preset", default="baselines/core", help="Preset name")
    parser.add_argument("--n", type=int, default=5, help="Number of trials")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    n = args.n
    outdir = (
        args.outdir
        or Path("runs")
        / datetime.utcnow().strftime("%Y%m%d")
        / args.preset.replace("/", "_")
        / args.suite
    )
    evaluate(args.suite, n, args.seed, args.preset, outdir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
