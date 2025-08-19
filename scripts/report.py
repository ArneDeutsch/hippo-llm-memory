"""Aggregate evaluation metrics into Markdown tables and plots.

This script scans baseline evaluation runs under
``runs/<date>/baselines/`` and summarises the metrics for each
suite/preset pair.  A short Markdown report is written to
``reports/<date>/baseline_summary.md``.  If ``matplotlib`` is available a
set of bar plots is produced alongside the report.

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

    Parameters
    ----------
    base:
        Path pointing to ``runs/<date>/baselines``.
    """

    data: Dict[Tuple[str, str], list[MetricDict]] = defaultdict(list)
    pattern = "*/*/*/metrics.json"
    for metrics_path in base.glob(pattern):
        run_dir = metrics_path.parent
        suite = run_dir.parent.name
        preset = run_dir.parent.parent.name
        try:
            with metrics_path.open() as fh:
                metrics = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - file is corrupt
            log.warning("failed to parse %s: %s", metrics_path, exc)
            continue
        if suite not in metrics:
            log.warning("suite %s missing in %s", suite, metrics_path)
            continue
        data[(suite, preset)].append(metrics[suite])
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


def render_markdown(summary: Summary) -> str:
    """Return a Markdown representation of ``summary``."""

    lines: list[str] = ["# Baseline Summary", ""]
    for suite in sorted(summary):
        presets = summary[suite]
        if not presets:
            continue
        lines.append(f"## {suite}")
        metric_keys = sorted(next(iter(presets.values())).keys())
        header = "| Preset | " + " | ".join(metric_keys) + " |"
        sep = "|---" * (len(metric_keys) + 1) + "|"
        lines.extend([header, sep])
        for preset in sorted(presets):
            metrics = presets[preset]
            row = (
                "| " + preset + " | " + " | ".join(f"{metrics[m]:.3f}" for m in metric_keys) + " |"
            )
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def _render_plots(summary: Summary, out_dir: Path) -> None:
    """Render simple bar plots if ``matplotlib`` is available."""

    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib missing
        log.warning("matplotlib unavailable: %s", exc)
        return

    for suite, presets in summary.items():
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
            plt.savefig(out_dir / f"{suite}_{metric}.png")
            plt.close()


def write_report(summary: Summary, out_dir: Path, plots: bool) -> Path:
    """Write ``baseline_summary.md`` and optional plots to ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "baseline_summary.md"
    md_path.write_text(render_markdown(summary))
    if plots:
        _render_plots(summary, out_dir)
    return md_path


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
    runs_path = runs_root / date / "baselines"
    summary = summarise(collect_metrics(runs_path))
    out_dir = Path(args.out_dir) / date
    md_path = write_report(summary, out_dir, args.plots)
    log.info("wrote %s", md_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
