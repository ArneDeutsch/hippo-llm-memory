# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Aggregate run metrics into per-preset summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def summarize_runs(root: Path, out: Path) -> None:
    """Aggregate metrics under ``root`` and write summaries to ``out``."""
    rows_by_preset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for metrics_path in root.rglob("metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        preset = metrics.get("preset", "unknown")
        suite = metrics.get("suite")
        n = metrics.get("n")
        seed = metrics.get("seed")
        suite_metrics = metrics.get("metrics", {}).get(suite, {})
        em = suite_metrics.get("post_em", suite_metrics.get("pre_em", 0.0))
        f1 = suite_metrics.get("post_f1", suite_metrics.get("pre_f1", 0.0))
        compute = metrics.get("metrics", {}).get("compute", {})
        tokens = compute.get("total_tokens")
        time_ms_per_100 = compute.get("time_ms_per_100")
        meta_path = metrics_path.with_name("meta.json")
        model = "unknown"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            model = meta.get("model", {}).get("id", "unknown")
        rows_by_preset[preset].append(
            {
                "task": suite,
                "model": model,
                "n": n,
                "seed": seed,
                "em": em,
                "f1": f1,
                "tokens": tokens,
                "time_ms_per_100": time_ms_per_100,
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "model", "n", "seed", "em", "f1", "tokens", "time_ms_per_100"]
    for preset, rows in rows_by_preset.items():
        safe = preset.replace("/", "_")
        csv_path = out / f"{safe}_summary.csv"
        json_path = out / f"{safe}_summary.json"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI tool
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root directory containing run outputs")
    parser.add_argument("--out", type=Path, required=True, help="Directory to write summaries")
    args = parser.parse_args(argv)
    summarize_runs(args.root, args.out)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
