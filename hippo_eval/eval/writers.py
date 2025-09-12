# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Filesystem writers for evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def write_baseline_metrics(rows: List[Dict[str, float]], out_dir: Path) -> Path:
    """Write ``rows`` to ``out_dir/metrics.csv`` and a success flag."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    fieldnames = [
        "suite",
        "preset",
        "em_raw_mean",
        "em_raw_ci",
        "em_norm_mean",
        "em_norm_ci",
        "f1_mean",
        "f1_ci",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "baselines_ok.flag").write_text("ok\n", encoding="utf-8")
    return csv_path


def write_metrics(path: Path, metrics: Dict[str, object]) -> Path:
    """Write evaluation ``metrics`` JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh)
    return path


def write_meta(path: Path, meta: Dict[str, object]) -> Path:
    """Write metadata JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return path


def write_csv(
    path: Path, rows: List[Dict[str, object]], fieldnames: List[str] | None = None
) -> Path:
    """Write ``rows`` to ``path`` as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def ensure_run_dirs(root: Path) -> Dict[str, Path]:
    """Ensure ``root`` exists and return a mapping of created paths."""
    root.mkdir(parents=True, exist_ok=True)
    return {"root": root}


__all__ = [
    "write_baseline_metrics",
    "write_metrics",
    "write_meta",
    "write_csv",
    "ensure_run_dirs",
]
