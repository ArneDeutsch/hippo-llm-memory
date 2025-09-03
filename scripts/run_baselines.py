#!/usr/bin/env python3
"""Aggregate baseline metrics and persist summary with confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

from hippo_mem.utils import validate_run_id  # noqa: E402


def _ci95(values: List[float]) -> float:
    """Return 95% confidence interval for ``values``.

    Parameters
    ----------
    values:
        Sample of metric values.
    """

    if len(values) < 2:  # pre: CI undefined for single sample
        return 0.0
    std = pstdev(values)
    return 1.96 * std / math.sqrt(len(values))


def collect_baseline_metrics(root: Path) -> List[Dict[str, float]]:
    """Collect per-suite baseline metrics under ``root``.

    Parameters
    ----------
    root:
        Directory ``runs/<run_id>/baselines`` containing per-seed outputs.
    """

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


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Root runs directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_id = validate_run_id(args.run_id)

    root = Path(args.runs_dir) / run_id / "baselines"
    rows = collect_baseline_metrics(root)
    write_metrics(rows, root)
    print(f"aggregated {len(rows)} baseline rows under {root}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
