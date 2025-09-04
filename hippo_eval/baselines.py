"""Baseline utilities for metrics aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def _ci95(values: List[float]) -> float:
    """Return 95% confidence interval for ``values``."""
    if len(values) < 2:
        return 0.0
    std = pstdev(values)
    return 1.96 * std / math.sqrt(len(values))


def aggregate_metrics(root: Path) -> List[Dict[str, float]]:
    """Collect per-suite baseline metrics under ``root``."""
    rows: List[Dict[str, float]] = []
    try:
        for preset_dir in root.iterdir():
            if not preset_dir.is_dir():
                continue
            preset = preset_dir.name
            for suite_dir in preset_dir.iterdir():
                if not suite_dir.is_dir():
                    continue
                suite = suite_dir.name
                em_raw_vals: List[float] = []
                em_norm_vals: List[float] = []
                f1_vals: List[float] = []
                for run_dir in suite_dir.iterdir():
                    metrics_path = run_dir / "metrics.json"
                    if not metrics_path.exists():
                        continue
                    with metrics_path.open("r", encoding="utf-8") as fh:
                        record = json.load(fh)
                    suite_metrics = record.get("metrics", {}).get(suite, {})
                    em_raw = suite_metrics.get("pre_em_raw")
                    em_norm = suite_metrics.get("pre_em_norm")
                    f1 = suite_metrics.get("pre_f1")
                    if None in (em_raw, em_norm, f1):
                        continue
                    em_raw_vals.append(float(em_raw))
                    em_norm_vals.append(float(em_norm))
                    f1_vals.append(float(f1))
                if em_raw_vals and em_norm_vals and f1_vals:
                    rows.append(
                        {
                            "suite": suite,
                            "preset": preset,
                            "em_raw_mean": mean(em_raw_vals),
                            "em_raw_ci": _ci95(em_raw_vals),
                            "em_norm_mean": mean(em_norm_vals),
                            "em_norm_ci": _ci95(em_norm_vals),
                            "f1_mean": mean(f1_vals),
                            "f1_ci": _ci95(f1_vals),
                        }
                    )
    except FileNotFoundError:
        rows = []
    if not rows:  # post: at least one row expected
        runs_dir = root.parent.parent
        candidates = (
            ", ".join(sorted(p.name for p in runs_dir.glob("*") if (p / "baselines").exists()))
            or "<none>"
        )
        msg = f"no baseline metrics under {root}; found: {candidates}"
        raise FileNotFoundError(msg)
    return rows
