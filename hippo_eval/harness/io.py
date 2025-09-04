"""Filesystem helpers for evaluation scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def write_metrics(rows: List[Dict[str, float]], out_dir: Path) -> Path:
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
