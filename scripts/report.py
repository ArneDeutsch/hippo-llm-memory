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
    """Collect metric dictionaries grouped by ``(suite, preset)``."""

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


def collect_retrieval(base: Path) -> Dict[str, list[dict[str, MetricDict]]]:
    """Collect retrieval telemetry grouped by suite."""

    data: Dict[str, list[dict[str, MetricDict]]] = defaultdict(list)
    for metrics_path in base.rglob("metrics.json"):
        parts = metrics_path.relative_to(base).parts
        if len(parts) < 3:
            continue
        suite = parts[-3]
        try:
            with metrics_path.open() as fh:
                record = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - file is corrupt
            log.warning("failed to parse %s: %s", metrics_path, exc)
            continue
        ret = record.get("retrieval")
        if ret:
            data[suite].append(ret)
    return data


def collect_gates(base: Path) -> Dict[str, list[dict[str, MetricDict]]]:
    """Collect gate telemetry grouped by suite."""

    data: Dict[str, list[dict[str, MetricDict]]] = defaultdict(list)
    for metrics_path in base.rglob("metrics.json"):
        parts = metrics_path.relative_to(base).parts
        if len(parts) < 3:
            continue
        suite = parts[-3]
        try:
            with metrics_path.open() as fh:
                record = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - file is corrupt
            log.warning("failed to parse %s: %s", metrics_path, exc)
            continue
        gates = record.get("gates")
        if gates:
            data[suite].append(gates)
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


def summarise_retrieval(
    data: Dict[str, list[dict[str, MetricDict]]],
) -> Dict[str, Dict[str, MetricDict]]:
    """Average retrieval stats per suite and memory."""

    summary: Dict[str, Dict[str, MetricDict]] = {}
    for suite, records in data.items():
        agg: Dict[str, Dict[str, float]] = {}
        for rec in records:
            for mem, stats in rec.items():
                dest = agg.setdefault(mem, {k: 0.0 for k in stats})
                for k, v in stats.items():
                    dest[k] += float(v)
        count = len(records)
        summary[suite] = {
            mem: {k: v / count for k, v in stats.items()} for mem, stats in agg.items()
        }
    return summary


def summarise_gates(
    data: Dict[str, list[dict[str, MetricDict]]],
) -> Dict[str, Dict[str, MetricDict]]:
    """Average gate stats per suite and memory."""

    summary: Dict[str, Dict[str, MetricDict]] = {}
    for suite, records in data.items():
        agg: Dict[str, Dict[str, float]] = {}
        for rec in records:
            for mem, stats in rec.items():
                dest = agg.setdefault(mem, {k: 0.0 for k in stats})
                for k, v in stats.items():
                    dest[k] += float(v)
        count = len(records)
        summary[suite] = {
            mem: {k: v / count for k, v in stats.items()} for mem, stats in agg.items()
        }
    return summary


def _render_markdown_suite(
    suite: str,
    presets: Dict[str, MetricDict],
    retrieval: Dict[str, MetricDict] | None,
    gates: Dict[str, MetricDict] | None,
) -> str:
    """Return a Markdown table for a single suite."""

    lines: list[str] = [f"# {suite} Summary", ""]
    if presets:
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
    if retrieval:
        lines.append("## Retrieval Telemetry")
        lines.append("| mem | requests | hit_rate_at_k | avg_latency_ms | tokens_returned |")
        lines.append("|---|---|---|---|---|")
        for mem in sorted(retrieval):
            stats = retrieval[mem]
            row = (
                f"| {mem} | {int(stats['requests'])} | {stats['hit_rate_at_k']:.3f} | "
                f"{stats['avg_latency_ms']:.3f} | {int(stats['tokens_returned'])} |"
            )
            lines.append(row)
        lines.append("")
    if gates:
        lines.append("## Gate Telemetry")
        lines.append(
            "| mem | attempts | inserted | aggregated | routed_to_episodic/blocked_new_edges |"
        )
        lines.append("|---|---|---|---|---|")
        for mem in sorted(gates):
            stats = gates[mem]
            key = "routed_to_episodic" if mem == "relational" else "blocked_new_edges"
            row = (
                f"| {mem} | {int(stats['attempts'])} | {int(stats['inserted'])} | "
                f"{int(stats['aggregated'])} | {int(stats.get(key, 0))} |"
            )
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


def write_reports(
    summary: Summary,
    retrieval: Dict[str, Dict[str, MetricDict]],
    gates: Dict[str, Dict[str, MetricDict]],
    out_dir: Path,
    plots: bool,
) -> Dict[str, Path]:
    """Write per-suite reports and optional plots to ``out_dir``."""

    paths: Dict[str, Path] = {}
    for suite, presets in summary.items():
        suite_dir = out_dir / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        md_path = suite_dir / "summary.md"
        md_path.write_text(
            _render_markdown_suite(suite, presets, retrieval.get(suite), gates.get(suite))
        )
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
    metric_data = collect_metrics(runs_path)
    retrieval_data = collect_retrieval(runs_path)
    gate_data = collect_gates(runs_path)
    summary = summarise(metric_data)
    retrieval_summary = summarise_retrieval(retrieval_data)
    gate_summary = summarise_gates(gate_data)
    out_dir = Path(args.out_dir) / date
    paths = write_reports(summary, retrieval_summary, gate_summary, out_dir, args.plots)
    for suite, md_path in paths.items():
        log.info("wrote %s for %s", md_path, suite)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
