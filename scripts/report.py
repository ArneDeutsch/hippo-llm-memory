"""Aggregate evaluation metrics into Markdown tables and plots.

This script scans evaluation runs under ``runs/<date>/`` and summarises
the metrics for each suite/preset pair.  For every suite a Markdown
report is written to ``reports/<date>/<suite>/summary.md`` and, if
``matplotlib`` is available, accompanying bar plots are produced.  The
reports include compute and memory columns when present in the metrics
files.  A top-level roll-up ``reports/<date>/index.md`` is also emitted
with an overall matrix table and links to per-suite summaries.

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
from statistics import mean, stdev
from typing import Dict, Iterable, Tuple

log = logging.getLogger(__name__)

MetricDict = Dict[str, float]
DISPLAY_NAMES = {
    "em_norm": "EM (norm)",
    "em_raw": "EM (raw)",
    "em": "EM",
    "pre_em": "EM (pre)",
    "post_em": "EM (post)",
    "delta_em": "ΔEM",
    "pre_f1": "F1 (pre)",
    "post_f1": "F1 (post)",
    "delta_f1": "ΔF1",
    "overlong": "overlong",
    "format_violation": "format_violation",
}
MetricStats = Dict[str, tuple[float, float]]
Summary = Dict[str, Dict[str, Dict[int, MetricStats]]]


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


def collect_metrics(
    base: Path,
) -> Dict[Tuple[str, str, int], Iterable[MetricDict]]:
    """Collect metric dictionaries grouped by ``(suite, preset, size)``."""

    data: Dict[Tuple[str, str, int], list[MetricDict]] = defaultdict(list)
    for metrics_path in base.rglob("metrics.json"):
        parts = metrics_path.relative_to(base).parts
        # expect: <preset...>/<suite>/<size>_<seed>/metrics.json
        if len(parts) < 4:
            continue
        suite = parts[-3]
        size_seed = parts[-2]
        try:
            size = int(size_seed.split("_")[0])
        except ValueError:
            log.warning("could not parse size from %s", metrics_path)
            continue
        preset = "/".join(parts[:-3]) or "unknown"
        try:
            with metrics_path.open() as fh:
                record = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - file is corrupt
            log.warning("failed to parse %s: %s", metrics_path, exc)
            continue
        metrics = record.get("metrics", {})
        diagnostics = record.get("diagnostics", {})
        if suite not in metrics:
            log.warning("suite %s missing in %s", suite, metrics_path)
            continue
        suite_metrics = metrics[suite]
        cleaned: Dict[str, float] = dict(suite_metrics)
        diag = diagnostics.get(suite, {})
        n_items = record.get("n", 0) or 0
        for key, val in diag.items():
            if n_items:
                cleaned[key] = float(val) / n_items
            else:
                cleaned[key] = float(val)
        # derive delta_* metrics and normalise solitary pre_* keys
        for key in list(cleaned.keys()):
            if key.startswith("post_"):
                base_key = key[5:]
                pre_key = f"pre_{base_key}"
                if pre_key in cleaned:
                    cleaned[f"delta_{base_key}"] = cleaned[key] - cleaned[pre_key]
            elif key.startswith("pre_") and f"post_{key[4:]}" not in cleaned:
                base_key = key[4:]
                cleaned[base_key] = cleaned.pop(key)
        compute_metrics = metrics.get("compute", {})
        data[(suite, preset, size)].append({**cleaned, **compute_metrics})
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


def collect_gates(base: Path) -> Dict[str, Dict[str, list[dict[str, MetricDict]]]]:
    """Collect gate telemetry grouped by suite and gate status."""

    data: Dict[str, Dict[str, list[dict[str, MetricDict]]]] = defaultdict(lambda: defaultdict(list))
    for metrics_path in base.rglob("metrics.json"):
        parts = metrics_path.relative_to(base).parts
        if len(parts) < 3:
            continue
        suite = parts[-3]
        status = "on"
        for p in parts:
            low = p.lower()
            if "gate" in low and ("off" in low or "false" in low):
                status = "off"
                break
            if "gate" in low and ("on" in low or "true" in low):
                status = "on"
                break
        try:
            with metrics_path.open() as fh:
                record = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - file is corrupt
            log.warning("failed to parse %s: %s", metrics_path, exc)
            continue
        gates = record.get("gates")
        if gates:
            data[suite][status].append(gates)
    return data


def summarise(data: Dict[Tuple[str, str, int], Iterable[MetricDict]]) -> Summary:
    """Average metrics for each ``(suite, preset, size)`` with 95% CI."""

    summary: Summary = {}
    for (suite, preset, size), metrics_list in data.items():
        if not metrics_list:
            continue
        # union of metric keys across records to be robust to missing fields
        keys: set[str] = set()
        for rec in metrics_list:
            keys.update(rec.keys())
        agg: Dict[str, tuple[float, float]] = {}
        for key in keys:
            vals = [m.get(key, 0.0) for m in metrics_list]
            count = len(vals)
            mval = mean(vals)
            sval = stdev(vals) if count > 1 else 0.0
            ci = 1.96 * sval / (count**0.5) if count > 1 else 0.0
            agg[key] = (mval, ci)
        summary.setdefault(suite, {}).setdefault(preset, {})[size] = agg
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
    data: Dict[str, Dict[str, list[dict[str, MetricDict]]]],
) -> Dict[str, Dict[str, Dict[str, MetricDict]]]:
    """Average gate stats per suite, status and memory with derived metrics."""

    summary: Dict[str, Dict[str, Dict[str, MetricDict]]] = {}
    for suite, variants in data.items():
        summary[suite] = {}
        for status, records in variants.items():
            agg: Dict[str, Dict[str, float]] = {}
            for rec in records:
                for mem, stats in rec.items():
                    dest = agg.setdefault(mem, {k: 0.0 for k in stats})
                    for k, v in stats.items():
                        dest[k] += float(v)
            count = len(records)
            mem_summary: Dict[str, MetricDict] = {
                mem: {k: v / count for k, v in stats.items()} for mem, stats in agg.items()
            }
            for mem, stats in mem_summary.items():
                attempts = stats.get("attempts", 0.0)
                if mem == "relational" and attempts:
                    stats["duplicate_rate"] = stats.get("aggregated", 0.0) / attempts
                if mem == "spatial" and attempts:
                    stats["nodes_per_1k"] = stats.get("nodes_added", 0.0) / attempts * 1000.0
                    stats["edges_per_1k"] = stats.get("edges_added", 0.0) / attempts * 1000.0
            summary[suite][status] = mem_summary
    return summary


def _aggregate_gates(
    summary: Dict[str, Dict[str, Dict[str, MetricDict]]],
) -> Dict[str, Dict[str, MetricDict]]:
    """Average gate stats across suites for index reporting."""

    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for variants in summary.values():
        for status, mems in variants.items():
            for mem, stats in mems.items():
                dest = agg.setdefault(status, {}).setdefault(mem, defaultdict(float))
                counts.setdefault(status, {}).setdefault(mem, 0)
                for k, v in stats.items():
                    dest[k] += v
                counts[status][mem] += 1
    result: Dict[str, Dict[str, MetricDict]] = {}
    for status, mems in agg.items():
        result[status] = {}
        for mem, stats in mems.items():
            cnt = counts[status][mem]
            result[status][mem] = {k: v / cnt for k, v in stats.items()}
    return result


def write_smoke(data_root: Path, out_path: Path, n_rows: int = 3) -> Path:
    """Write sample rows from each suite to ``out_path``.

    The function looks for the smallest ``*.jsonl`` dataset file in every
    ``data/<suite>`` directory, extracts up to ``n_rows`` rows, and renders
    them as a Markdown table.  The resulting file is intended for quick
    human inspection of prompt/answer formats.

    Parameters
    ----------
    data_root:
        Root directory containing per-suite datasets.
    out_path:
        Path of the Markdown file to write.
    n_rows:
        Maximum number of rows to sample per suite.
    """

    lines: list[str] = ["# Smoke Samples", ""]
    for suite_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        candidates = sorted(suite_dir.glob("*.jsonl"))
        if not candidates:
            continue

        def size_key(p: Path) -> int:
            try:
                return int(p.stem.split("_")[0])
            except ValueError:  # pragma: no cover - file name mismatch
                return 10**9

        sample_file = min(candidates, key=size_key)
        rows: list[tuple[str, str]] = []
        with sample_file.open() as fh:
            for _ in range(n_rows):
                line = fh.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = str(obj.get("prompt") or obj.get("question") or "")
                answer = str(obj.get("answer"))
                rows.append((prompt, answer))
        if not rows:
            continue
        lines.extend([f"## {suite_dir.name}", "", "| prompt | answer |", "|---|---|"])
        for prompt, answer in rows:
            p = prompt.replace("|", "\\|")
            a = answer.replace("|", "\\|")
            lines.append(f"| {p} | {a} |")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def _render_markdown_suite(
    suite: str,
    presets: Dict[str, Dict[int, MetricStats]],
    retrieval: Dict[str, MetricDict] | None,
    gates: Dict[str, Dict[str, MetricDict]] | None,
) -> str:
    """Return a Markdown table for a single suite."""

    lines: list[str] = [f"# {suite} Summary", ""]
    if presets:
        metric_keys = sorted(
            {
                key
                for preset in presets.values()
                for size_stats in preset.values()
                for key in size_stats
            }
        )
        preferred = [
            "pre_em",
            "post_em",
            "delta_em",
            "em_raw",
            "em_norm",
            "em",
            "pre_f1",
            "post_f1",
            "delta_f1",
            "f1",
            "overlong",
            "format_violation",
        ]
        ordered = [k for k in preferred if k in metric_keys] + [
            k for k in metric_keys if k not in preferred
        ]
        display = [DISPLAY_NAMES.get(k, k) for k in ordered]
        header = "| Preset | Size | " + " | ".join(display) + " |"
        sep = "|---" * (len(ordered) + 2) + "|"
        lines.extend([header, sep])
        for preset in sorted(presets):
            sizes = presets[preset]
            for size in sorted(sizes):
                metrics = sizes[size]
                vals: list[str] = []
                for key in ordered:
                    stat = metrics.get(key)
                    if stat is None:
                        vals.append("–")
                    else:
                        mval, ci = stat
                        vals.append(f"{mval:.3f} ± {ci:.3f}")
                row = f"| {preset} | {size} | " + " | ".join(vals) + " |"
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
        int_keys = {
            "attempts",
            "inserted",
            "aggregated",
            "routed_to_episodic",
            "blocked_new_edges",
            "nodes_added",
            "edges_added",
        }
        order = [
            "attempts",
            "inserted",
            "aggregated",
            "duplicate_rate",
            "nodes_added",
            "edges_added",
            "nodes_per_1k",
            "edges_per_1k",
            "routed_to_episodic",
            "blocked_new_edges",
        ]
        for status, mems in gates.items():
            lines.append(f"### Gate {status.upper()}")
            metric_keys = [k for k in order if any(k in s for s in mems.values())]
            header = "| mem | " + " | ".join(metric_keys) + " |"
            sep = "|---" * (len(metric_keys) + 1) + "|"
            lines.extend([header, sep])
            for mem in sorted(mems):
                stats = mems[mem]
                vals: list[str] = []
                for key in metric_keys:
                    val = stats.get(key, 0)
                    if key in int_keys:
                        vals.append(str(int(val)))
                    else:
                        vals.append(f"{val:.3f}")
                lines.append("| " + mem + " | " + " | ".join(vals) + " |")
            lines.append("")
        if "on" in gates and "off" in gates:
            lines.append("### Gate ON vs OFF")
            lines.append("| mem | metric | on | off | Δ |")
            lines.append("|---|---|---|---|---|")
            on = gates["on"]
            off = gates["off"]
            rel_on = on.get("relational")
            rel_off = off.get("relational")
            if rel_on and rel_off and "duplicate_rate" in rel_on and "duplicate_rate" in rel_off:
                dv = rel_on["duplicate_rate"] - rel_off["duplicate_rate"]
                lines.append(
                    f"| relational | duplicate_rate | {rel_on['duplicate_rate']:.3f} | "
                    f"{rel_off['duplicate_rate']:.3f} | {dv:+.3f} |"
                )
            spa_on = on.get("spatial")
            spa_off = off.get("spatial")
            if spa_on and spa_off:
                for metric in ["nodes_per_1k", "edges_per_1k"]:
                    if metric in spa_on and metric in spa_off:
                        dv = spa_on[metric] - spa_off[metric]
                        lines.append(
                            f"| spatial | {metric} | {spa_on[metric]:.3f} | "
                            f"{spa_off[metric]:.3f} | {dv:+.3f} |"
                        )
            lines.append("")
    return "\n".join(lines)


def _render_plots_suite(
    suite: str, presets: Dict[str, Dict[int, MetricStats]], out_dir: Path
) -> None:
    """Render simple bar plots for one suite if ``matplotlib`` is available."""

    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib missing
        log.warning("matplotlib unavailable: %s", exc)
        return

    metric_keys = sorted(
        {key for preset in presets.values() for size_stats in preset.values() for key in size_stats}
    )
    for metric in metric_keys:
        labels: list[str] = []
        values: list[float] = []
        for preset in sorted(presets):
            for size in sorted(presets[preset]):
                labels.append(f"{preset}-{size}")
                stats = presets[preset][size].get(metric)
                values.append(stats[0] if stats else 0.0)
        plt.figure()
        plt.bar(labels, values)
        plt.ylabel(metric)
        plt.title(f"{suite} {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"{metric}.png")
        plt.close()


def _write_index(
    summary: Summary,
    suite_paths: Dict[str, Path],
    gates: Dict[str, Dict[str, Dict[str, MetricDict]]],
    out_dir: Path,
) -> Path:
    """Write a top-level roll-up report and return its path."""

    rollup: list[tuple[str, str, Dict[str, float]]] = []
    metric_keys: set[str] = set()
    em_by_preset: Dict[str, list[float]] = defaultdict(list)
    for suite, presets in summary.items():
        for preset, sizes in presets.items():
            agg: Dict[str, float] = {}
            for key in {k for stats in sizes.values() for k in stats}:
                vals = [stats[key][0] for stats in sizes.values() if key in stats]
                agg[key] = sum(vals) / len(vals)
            rollup.append((suite, preset, agg))
            metric_keys.update(agg.keys())
            em_key = (
                "em_norm"
                if "em_norm" in agg
                else ("em" if "em" in agg else ("em_raw" if "em_raw" in agg else None))
            )
            if em_key:
                em_by_preset[preset].append(agg[em_key])

    metric_keys = sorted(metric_keys)
    preferred = [
        "pre_em",
        "post_em",
        "delta_em",
        "em_raw",
        "em_norm",
        "em",
        "pre_f1",
        "post_f1",
        "delta_f1",
        "f1",
        "overlong",
        "format_violation",
    ]
    ordered = [k for k in preferred if k in metric_keys] + [
        k for k in metric_keys if k not in preferred
    ]
    display = [DISPLAY_NAMES.get(k, k) for k in ordered]
    lines: list[str] = ["# Overall Summary", ""]
    if ordered:
        header = "| Suite | Preset | " + " | ".join(display) + " |"
        sep = "|---" * (len(ordered) + 2) + "|"
        lines.extend([header, sep])
        for suite, preset, metrics in rollup:
            vals = [f"{metrics.get(k, float('nan')):.3f}" if k in metrics else "–" for k in ordered]
            lines.append(f"| {suite} | {preset} | " + " | ".join(vals) + " |")
        lines.append("")

    assets_dir = out_dir / "assets"
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt

        if em_by_preset:
            presets = sorted(em_by_preset)
            values = [sum(em_by_preset[p]) / len(em_by_preset[p]) for p in presets]
            assets_dir.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.bar(presets, values)
            plt.ylabel("EM")
            plt.title("EM by preset")
            plt.tight_layout()
            plt.savefig(assets_dir / "overall_em.png")
            plt.close()
            lines.append("![Overall EM](assets/overall_em.png)")
            lines.append("")
    except Exception as exc:  # pragma: no cover - matplotlib missing
        log.warning("matplotlib unavailable: %s", exc)

    if gates:
        lines.append("## Gate Telemetry")
        agg = _aggregate_gates(gates)
        header = "| status | mem | duplicate_rate | nodes_per_1k | edges_per_1k |"
        lines.extend([header, "|---|---|---|---|---|"])
        for status, mems in sorted(agg.items()):
            for mem, stats in sorted(mems.items()):
                dr = stats.get("duplicate_rate", float("nan"))
                npk = stats.get("nodes_per_1k", float("nan"))
                epk = stats.get("edges_per_1k", float("nan"))
                lines.append(f"| {status} | {mem} | {dr:.3f} | {npk:.3f} | {epk:.3f} |")
        lines.append("")

    lines.append("## Per-suite summaries")
    for suite in sorted(suite_paths):
        lines.append(f"- [{suite}]({suite}/summary.md)")
    smoke = out_dir / "smoke.md"
    if smoke.exists():
        lines.append("")
        lines.append("See also: [smoke.md](smoke.md)")

    index_path = out_dir / "index.md"
    index_path.write_text("\n".join(lines))
    return index_path


def write_reports(
    summary: Summary,
    retrieval: Dict[str, Dict[str, MetricDict]],
    gates: Dict[str, Dict[str, Dict[str, MetricDict]]],
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
    idx = _write_index(summary, paths, gates, out_dir)
    log.info("wrote %s", idx)
    return paths


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="directory containing run outputs")
    parser.add_argument("--out-dir", default="reports", help="directory to write reports to")
    parser.add_argument("--data-dir", default="data", help="dataset directory for smoke report")
    parser.add_argument(
        "--date", default=None, help="run date in YYYYMMDD format; latest if omitted"
    )
    parser.add_argument("--plots", action="store_true", help="render bar plots using matplotlib")
    parser.add_argument("--smoke", action="store_true", help="also write smoke.md with sample rows")
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
    if args.smoke:
        write_smoke(Path(args.data_dir), out_dir / "smoke.md")
    paths = write_reports(summary, retrieval_summary, gate_summary, out_dir, args.plots)
    for suite, md_path in paths.items():
        log.info("wrote %s for %s", md_path, suite)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
