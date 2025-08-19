"""Aggregate evaluation metrics into Markdown tables and plots.

This script scans evaluation runs under ``runs/<date>/`` and summarises
the metrics for each suite/preset pair.  For every suite a Markdown
report is written to ``reports/<date>/<suite>/summary.md`` and, if
``matplotlib`` is available, accompanying bar plots are produced.  The
reports include compute and memory columns when present in the metrics
files.

Example usage from the command line::

    python scripts/report.py --date 20250101

The script assumes the directory layout created by
:mod:`scripts.eval_bench`.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Tuple

log = logging.getLogger(__name__)

MetricDict = Dict[str, float]
Summary = Dict[str, Dict[str, MetricDict]]


def _find_latest_date(root: Path) -> str:
    """Return the latest run date available under ``root``.

    Parameters
    ----------
    root:
        Directory containing dated subdirectories.
    """

    dates = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not dates:
        msg = f"no run directories found under {root}"
        raise FileNotFoundError(msg)
    return dates[-1]


def collect_metrics(base: Path) -> Dict[Tuple[str, str], Iterable[MetricDict]]:
    """Collect metric dictionaries grouped by ``(suite, preset)``.

    Each ``metrics.json`` is expected to contain a ``{"metrics": {...}}`` object
    where the suite metrics live under ``record["metrics"][suite]`` and optional
    compute metrics are provided under ``record["metrics"]["compute"]``. Compute
    metrics are merged with the suite metrics.

    Parameters
    ----------
    base:
        Path pointing to ``runs/<date>``.
    """

    data: Dict[Tuple[str, str], list[MetricDict]] = defaultdict(list)
    for metrics_path in base.rglob("metrics.json"):
        parts = metrics_path.relative_to(base).parts
        if len(parts) < 3:
            continue
        suite = parts[-3]
        preset = "/".join(parts[:-3]) or "unknown"
        try:
            with metrics_path.open() as fh:
                record = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - file is corrupt
            log.warning("failed to parse %s: %s", metrics_path, exc)
            continue
        metrics = record.get("metrics", {})
        if suite not in metrics:
            log.warning("suite %s missing in %s", suite, metrics_path)
            continue
        suite_metrics = metrics[suite]
        compute_metrics = metrics.get("compute", {})
        data[(suite, preset)].append({**suite_metrics, **compute_metrics})
    return data


def summarise(data: Dict[Tuple[str, str], Iterable[MetricDict]]) -> Summary:
    """Average metrics for each suite/preset pair."""

    summary: Summary = {}
    for (suite, preset), metrics_list in data.items():
        if not metrics_list:
            continue
        agg = {k: mean(m[k] for m in metrics_list) for k in metrics_list[0]}
        summary.setdefault(suite, {})[preset] = agg
    return summary


def _render_markdown_suite(suite: str, presets: Dict[str, MetricDict]) -> str:
    """Return a Markdown table for a single suite."""

    lines: list[str] = [f"# {suite} Summary", ""]
    if not presets:
        return "\n".join(lines)
    metric_keys = sorted(next(iter(presets.values())).keys())
    header = "| Preset | " + " | ".join(metric_keys) + " |"
    sep = "|---" * (len(metric_keys) + 1) + "|"
    lines.extend([header, sep])
    for preset in sorted(presets):
        metrics = presets[preset]
        row = "| " + preset + " | " + " | ".join(f"{metrics[m]:.3f}" for m in metric_keys) + " |"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def _render_plots_suite(suite: str, presets: Dict[str, MetricDict], out_dir: Path) -> None:
    """Render simple bar plots for one suite if ``matplotlib`` is available."""

    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib missing
        log.warning("matplotlib unavailable: %s", exc)
        return

    metric_keys = sorted(next(iter(presets.values())).keys())
    for metric in metric_keys:
        labels = list(presets.keys())
        values = [presets[p][metric] for p in labels]
        plt.figure()
        plt.bar(labels, values)
        plt.ylabel(metric)
        plt.title(f"{suite} {metric}")
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"{metric}.png")
        plt.close()


def write_reports(summary: Summary, out_dir: Path, plots: bool) -> Dict[str, Path]:
    """Write per-suite reports and optional plots to ``out_dir``.

    Returns a mapping from suite name to the written Markdown path.
    """

    paths: Dict[str, Path] = {}
    for suite, presets in summary.items():
        suite_dir = out_dir / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        md_path = suite_dir / "summary.md"
        md_path.write_text(_render_markdown_suite(suite, presets))
        paths[suite] = md_path
        if plots:
            _render_plots_suite(suite, presets, suite_dir)
    return paths


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="directory containing run outputs")
    parser.add_argument("--out-dir", default="reports", help="directory to write reports to")
    parser.add_argument(
        "--date", default=None, help="run date in YYYYMMDD format; latest if omitted"
    )
    parser.add_argument("--plots", action="store_true", help="render bar plots using matplotlib")
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)
    date = args.date or _find_latest_date(runs_root)
    runs_path = runs_root / date
    summary = summarise(collect_metrics(runs_path))
    out_dir = Path(args.out_dir) / date
    paths = write_reports(summary, out_dir, args.plots)
    for suite, md_path in paths.items():
        log.info("wrote %s for %s", md_path, suite)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
