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
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, Tuple

from hippo_mem.common.telemetry import validate_retrieval_snapshot

log = logging.getLogger(__name__)

MetricDict = Dict[str, float]
MetricStat = tuple[float, float] | None
DISPLAY_NAMES = {
    "em_norm": "EM (norm)",
    "em_raw": "EM (raw)",
    "em": "EM",
    "pre_em": "EM (pre)",
    "post_em": "EM (post)",
    "delta_em": "ΔEM",
    "pre_em_norm": "EM (pre, norm)",
    "post_em_norm": "EM (post, norm)",
    "delta_em_norm": "ΔEM (norm)",
    "pre_f1": "F1 (pre)",
    "post_f1": "F1 (post)",
    "delta_f1": "ΔF1",
    "overlong": "overlong",
    "format_violation": "format_violation",
}
MetricStats = Dict[str, MetricStat]
Summary = Dict[str, Dict[str, Dict[int, MetricStats]]]


def load_metrics(path: Path) -> dict:
    """Load a ``metrics.json`` file with backward compatibility."""

    with path.open() as fh:
        record = json.load(fh)
    if "version" not in record:
        record["version"] = 1
    if "gating" not in record and "gates" in record:
        record["gating"] = record.pop("gates")
    return record


def _format_stat(stat: tuple[float, float]) -> str:
    """Return a formatted statistic with 95% CI.

    Parameters
    ----------
    stat:
        Tuple of ``(mean, ci)`` where ``ci`` may be zero when only a single
        seed contributed.
    """

    mean_val, ci = stat
    return f"{mean_val:.3f} ± {ci:.3f}"


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
            record = load_metrics(metrics_path)
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
        store = record.get("store")
        if isinstance(store, dict):
            cleaned["store_size"] = float(store.get("size", 0))
        diag = diagnostics.get(suite, {})
        n_items = record.get("n", 0) or 0
        for key, val in diag.items():
            if val is None:
                continue
            if n_items:
                cleaned[key] = float(val) / n_items
            else:
                cleaned[key] = float(val)
        retrieval = record.get("retrieval", {})
        for mem, stats in retrieval.items():
            cleaned[f"retrieval_{mem}_requests"] = float(stats.get("requests", 0))
        gating = record.get("gating") or record.get("gates", {})
        if isinstance(gating, dict):
            cleaned["gate_attempts"] = float(
                sum(float(m.get("attempts", 0)) for m in gating.values())
            )
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


def _missing_post_metrics(
    metric_data: Dict[Tuple[str, str, int], Iterable[MetricDict]],
) -> list[Tuple[str, str, int]]:
    """Return groups lacking any ``post_*`` keys."""

    missing: list[Tuple[str, str, int]] = []
    for key, runs in metric_data.items():
        for record in runs:
            if not any(k.startswith("post_") for k in record):
                missing.append(key)
                break
    return missing


def collect_retrieval(base: Path) -> Dict[str, list[dict[str, MetricDict]]]:
    """Collect retrieval telemetry grouped by suite."""

    data: Dict[str, list[dict[str, MetricDict]]] = defaultdict(list)
    for metrics_path in base.rglob("metrics.json"):
        parts = metrics_path.relative_to(base).parts
        if len(parts) < 3:
            continue
        suite = parts[-3]
        try:
            record = load_metrics(metrics_path)
        except json.JSONDecodeError as exc:  # pragma: no cover - file is corrupt
            log.warning("failed to parse %s: %s", metrics_path, exc)
            continue
        ret = record.get("retrieval")
        if isinstance(ret, dict):
            for stats in ret.values():
                validate_retrieval_snapshot(stats, strict=True)
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
            record = load_metrics(metrics_path)
        except json.JSONDecodeError as exc:  # pragma: no cover - file is corrupt
            log.warning("failed to parse %s: %s", metrics_path, exc)
            continue
        gates = record.get("gating")
        if isinstance(gates, dict):
            data[suite][status].append(gates)
    return data


def collect_gate_ablation(base: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collect metrics for gate ON/OFF ablations.

    Returns a mapping ``memory -> {status -> {metric: value}}`` capturing
    ``store_size``, ``accepted`` and ``pre_em`` for each gate status.
    """

    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metrics_path in base.rglob("metrics.json"):
        parts = metrics_path.relative_to(base).parts
        if "gate_on" in parts:
            status = "on"
        elif "gate_off" in parts:
            status = "off"
        else:
            continue
        suite = parts[-3] if len(parts) >= 3 else ""
        mem = {
            "episodic": "episodic",
            "semantic": "relational",
            "spatial": "spatial",
        }.get(suite)
        if mem is None:
            continue
        try:
            record = load_metrics(metrics_path)
        except json.JSONDecodeError:
            continue
        metrics = record.get("metrics", {}).get(suite, {})
        gates = (record.get("gating") or {}).get(mem, {})
        store_size = float(record.get("store", {}).get("size", 0))
        em = float(metrics.get("pre_em", metrics.get("em", 0.0)))
        entry = data.setdefault(mem, {}).setdefault(status, {})
        entry["store_size"] = store_size
        entry["accepted"] = float(gates.get("accepted", 0))
        entry["em"] = em
    return data


def collect_lineage(base: Path) -> Dict[str, Dict[str, set]]:
    """Collect lineage fields grouped by suite.

    The returned mapping includes sets for ``seeds``, ``sizes``,
    ``profiles`` (dataset profiles), ``replay_samples`` and ``store_source``.
    """

    data: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for metrics_path in base.rglob("metrics.json"):
        try:
            record = load_metrics(metrics_path)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            continue
        suite = record.get("suite")
        if not suite:
            continue
        entry = data[suite]
        seed = record.get("seed")
        if seed is not None:
            entry.setdefault("seeds", set()).add(int(seed))
        size = record.get("n")
        if size is not None:
            entry.setdefault("sizes", set()).add(int(size))
        profile = record.get("dataset_profile") or "default"
        entry.setdefault("profiles", set()).add(str(profile))
        replay_samples = record.get("replay", {}).get("samples")
        if replay_samples is not None:
            entry.setdefault("replay_samples", set()).add(int(replay_samples))
        store_source = record.get("store", {}).get("source")
        if store_source:
            entry.setdefault("store_source", set()).add(str(store_source))
    return data


def summarise(
    data: Dict[Tuple[str, str, int], Iterable[MetricDict]],
) -> tuple[Summary, Dict[str, tuple[int, int]]]:
    """Average metrics for each ``(suite, preset, size)`` with 95% CI.

    Returns the summary and a mapping of ``suite`` to
    ``(missing_pre, total_pre)`` counts so callers can guard against large
    gaps in pre-phase metrics.
    """

    summary: Summary = {}
    missing_pre: Dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for (suite, preset, size), metrics_list in data.items():
        if not metrics_list:
            continue
        # union of metric keys across records to be robust to missing fields
        keys: set[str] = set()
        for rec in metrics_list:
            rec_keys = set(rec.keys())
            keys.update(rec_keys)
            for key in rec_keys:
                if key.startswith("post_"):
                    keys.add(f"pre_{key[5:]}")
        agg: Dict[str, MetricStat] = {}
        for key in keys:
            vals = [m[key] for m in metrics_list if key in m and m[key] is not None]
            missing = len(metrics_list) - len(vals)
            if key.startswith("pre_"):
                miss, total = missing_pre[suite]
                missing_pre[suite] = (miss + missing, total + len(metrics_list))
            if not vals:
                agg[key] = None
                continue
            count = len(vals)
            mval = mean(vals)
            sval = stdev(vals) if count > 1 else 0.0
            ci = 1.96 * sval / (count**0.5) if count > 1 else 0.0
            agg[key] = (mval, ci)
        summary.setdefault(suite, {}).setdefault(preset, {})[size] = agg
    return summary, missing_pre


def _missing_pre_suites(
    missing_counts: Dict[str, tuple[int, int]], threshold: float = 0.2
) -> list[str]:
    """Return suites where ``pre_*`` metrics are missing beyond ``threshold``.

    Parameters
    ----------
    missing_counts:
        Mapping of suite to ``(missing, total)`` counts produced by
        :func:`summarise`.
    threshold:
        Fraction of missing values above which suites should be flagged.
    """

    bad: list[str] = []
    for suite, (missing, total) in missing_counts.items():
        if total and missing / total > threshold:
            bad.append(suite)
    return bad


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
    seed_count: int,
    lineage: Dict[str, set] | None = None,
) -> str:
    """Return a Markdown table for a single suite."""

    lines: list[str] = [f"# {suite} Summary", ""]
    if lineage:
        pills: list[str] = []
        seeds = lineage.get("seeds")
        if seeds:
            seeds_str = ",".join(str(s) for s in sorted(seeds))
            pills.append(f"`seeds:{seeds_str}`")
        sizes = lineage.get("sizes")
        if sizes:
            sizes_str = ",".join(str(s) for s in sorted(sizes))
            pills.append(f"`sizes:{sizes_str}`")
        profiles = lineage.get("profiles")
        if profiles:
            prof_str = ",".join(sorted(profiles))
            pills.append(f"`profile:{prof_str}`")
        replay_samples = lineage.get("replay_samples")
        if replay_samples:
            rep_str = ",".join(str(s) for s in sorted(replay_samples))
            pills.append(f"`replay.samples:{rep_str}`")
        store_src = lineage.get("store_source")
        if store_src:
            src_str = ",".join(sorted(store_src))
            pills.append(f"`store:{src_str}`")
        if pills:
            lines.append("> " + " ".join(pills))
            lines.append("")
    if presets:
        saturated = False
        for preset_stats in presets.values():
            for size_stats in preset_stats.values():
                stat = size_stats.get("pre_em_norm") or size_stats.get("em_norm")
                if stat and stat[0] >= 0.98:
                    saturated = True
                    break
            if saturated:
                break
        if saturated:
            lines.extend(["> ⚠️ saturated: pre_em_norm ≥ 0.98", ""])

        metric_keys = sorted(
            {
                key
                for preset in presets.values()
                for size_stats in preset.values()
                for key in size_stats
                if not key.startswith("retrieval_") and not key.startswith("gate_")
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
        header = "| Preset | Size | " + " | ".join(display) + " | ⚠️ |"
        sep = "|---" * (len(ordered) + 3) + "|"
        lines.append("## Uplift")
        if seed_count <= 1:
            lines.extend(["> single-seed run: CI bands unavailable", ""])
        lines.extend([header, sep])
        rows: list[tuple[float, str, int, MetricStats]] = []
        baseline_pre = 0.0
        for preset, sizes in presets.items():
            if preset.startswith("baselines/"):
                first = next(iter(sizes.values()))
                baseline_pre = (first.get("pre_em_norm") or first.get("em_norm") or (0.0, 0.0))[0]
                break
        if suite == "semantic":
            for preset, sizes in presets.items():
                for size, metrics in sizes.items():
                    em_raw_stat = metrics.get("em_raw") or (0.0, 0.0)
                    rows.append((em_raw_stat[0], preset, size, metrics))
            rows.sort(key=lambda x: x[0], reverse=True)
        else:
            for preset in sorted(presets):
                sizes = presets[preset]
                for size in sorted(sizes):
                    rows.append((0.0, preset, size, sizes[size]))
        warnings_summary: list[str] = []
        for _, preset, size, metrics in rows:
            vals: list[str] = []
            warn: list[str] = []
            for key in ordered:
                stat = metrics.get(key)
                if stat is None:
                    if key.startswith("pre_"):
                        vals.append("_missing_")
                        if "MissingPre" not in warn:
                            warn.append("MissingPre")
                    else:
                        vals.append("–")
                else:
                    vals.append(_format_stat(stat))
            note: str = ""
            store_size_stat = metrics.get("store_size")
            store_size = store_size_stat[0] if store_size_stat else 0.0
            retrieval_reqs = [
                (metrics.get(f"retrieval_{mem}_requests") or (0.0, 0.0))[0]
                for mem in ("episodic", "relational", "spatial")
            ]
            gate_stat = metrics.get("gate_attempts")
            gate_attempts = gate_stat[0] if gate_stat else 0.0
            pre_norm_stat = metrics.get("pre_em_norm") or metrics.get("em_norm")
            pre_norm = pre_norm_stat[0] if pre_norm_stat else 0.0
            if preset.startswith("baselines/") and (
                store_size > 0 or any(r > 0 for r in retrieval_reqs)
            ):
                warn.append("BaselineTelemetry")
            if "no_retrieval" in preset and any(r > 0 for r in retrieval_reqs):
                warn.append("NoRetrievalTelemetry")
            if pre_norm >= 0.98 and baseline_pre < 0.20 and not preset.startswith("baselines/"):
                warn.append("SaturationSuspect")
            if (
                preset.startswith("memory/")
                and gate_attempts == 0
                and (store_size > 0 or any(r > 0 for r in retrieval_reqs))
            ):
                warn.append("GateNoOp")
            if warn:
                note = "⚠️ " + ", ".join(warn)
                warnings_summary.append(f"{preset} {size}: {', '.join(warn)}")
            row = f"| {preset} | {size} | " + " | ".join(vals) + " | " + note + " |"
            lines.append(row)
        lines.append("")
        if warnings_summary:
            lines.append("### Warnings")
            for w in warnings_summary:
                lines.append(f"- {w}")
            lines.append("")
    if retrieval:
        lines.append("## Retrieval Telemetry")
        note_path = (
            Path(__file__).resolve().parent.parent / "reports" / "templates" / "retrieval_note.md"
        )
        if note_path.exists():
            lines.append(note_path.read_text().strip())
        else:
            lines.append(
                "> Hits reflect actual recalled traces; cue-only fallbacks are excluded from telemetry."
            )
        lines.append(
            "| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for mem in sorted(retrieval):
            stats = retrieval[mem]
            row = (
                f"| {mem} | {int(stats.get('k', 0))} | {int(stats.get('batch_size', 0))} | "
                f"{int(stats['requests'])} | {int(stats.get('hits_at_k', stats.get('hits', 0)))} | "
                f"{stats['hit_rate_at_k']:.3f} | {int(stats['tokens_returned'])} | "
                f"{stats['avg_latency_ms']:.3f} |"
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
            lines.append("| mem | store_on | store_off | accepted_on | accepted_off | ΔEM |")
            lines.append("|---|---|---|---|---|---|")
            on = gates["on"]
            off = gates["off"]

            def _metric(status: str, metric: str) -> float:
                for preset_name, size_stats in presets.items():
                    if preset_name.endswith(f"gate_{status}"):
                        stats = next(iter(size_stats.values()))
                        return stats.get(metric, (0.0, 0.0))[0]
                return 0.0

            store_on = _metric("on", "store_size")
            store_off = _metric("off", "store_size")
            em_on = _metric("on", "pre_em") or _metric("on", "em")
            em_off = _metric("off", "pre_em") or _metric("off", "em")
            delta_em = em_on - em_off

            mems = sorted(set(on) | set(off))
            for mem in mems:
                accepted_on = on.get(mem, {}).get("accepted", 0)
                accepted_off = off.get(mem, {}).get("accepted", 0)
                lines.append(
                    f"| {mem} | {store_on:.0f} | {store_off:.0f} | {accepted_on:.0f} | {accepted_off:.0f} | {delta_em:+.3f} |"
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

    if "pre_em" in metric_keys and "post_em" in metric_keys:
        labels: list[str] = []
        pre_vals: list[float] = []
        pre_ci: list[float] = []
        post_vals: list[float] = []
        post_ci: list[float] = []
        for preset in sorted(presets):
            for size in sorted(presets[preset]):
                labels.append(f"{preset}-{size}")
                pre = presets[preset][size].get("pre_em")
                post = presets[preset][size].get("post_em")
                pre_vals.append(pre[0] if pre else 0.0)
                pre_ci.append(pre[1] if pre else 0.0)
                post_vals.append(post[0] if post else 0.0)
                post_ci.append(post[1] if post else 0.0)
        x = list(range(len(labels)))
        width = 0.4
        plt.figure()
        plt.bar(
            [i - width / 2 for i in x],
            pre_vals,
            width,
            yerr=pre_ci if any(pre_ci) else None,
            label="pre",
        )
        plt.bar(
            [i + width / 2 for i in x],
            post_vals,
            width,
            yerr=post_ci if any(post_ci) else None,
            label="post",
        )
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("EM")
        plt.title(f"{suite} uplift")
        plt.legend()
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "uplift.png")
        plt.close()


def _write_index(
    summary: Summary,
    suite_paths: Dict[str, Path],
    gates: Dict[str, Dict[str, Dict[str, MetricDict]]],
    gate_ablation: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
    seed_count: int,
) -> Path:
    """Write a top-level roll-up report and return its path."""

    rollup: list[tuple[str, str, Dict[str, float | None]]] = []
    metric_keys: set[str] = set()
    em_by_preset: Dict[str, list[float]] = defaultdict(list)
    for suite, presets in summary.items():
        for preset, sizes in presets.items():
            agg: Dict[str, float | None] = {}
            for key in {k for stats in sizes.values() for k in stats}:
                vals = [
                    stats[key][0]
                    for stats in sizes.values()
                    if key in stats and stats[key] is not None
                ]
                if vals:
                    agg[key] = sum(vals) / len(vals)
                else:
                    agg[key] = None
            rollup.append((suite, preset, agg))
            metric_keys.update(agg.keys())
            if suite == "semantic" and "em_raw" in agg:
                em_key = "em_raw"
            else:
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
    if seed_count >= 2:
        lines.extend([f"> aggregated over {seed_count} seeds (95% CI)", ""])
    else:
        lines.extend(["> single-seed run: CI bands unavailable", ""])
    if ordered:
        header = "| Suite | Preset | " + " | ".join(display) + " | ⚠️ |"
        sep = "|---" * (len(ordered) + 3) + "|"
        lines.extend([header, sep])
        for suite, preset, metrics in rollup:
            vals: list[str] = []
            warn: list[str] = []
            for k in ordered:
                stat = metrics.get(k)
                if stat is None:
                    if k.startswith("pre_"):
                        vals.append("_missing_")
                        if "MissingPre" not in warn:
                            warn.append("MissingPre")
                    else:
                        vals.append("–")
                else:
                    vals.append(f"{stat:.3f}")
            note = "⚠️ " + ", ".join(warn) if warn else ""
            lines.append(f"| {suite} | {preset} | " + " | ".join(vals) + " | " + note + " |")
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

    if gate_ablation:
        lines.append("## Gate ON vs OFF")
        lines.append("| mem | store_on | store_off | accepted_on | accepted_off | ΔEM |")
        lines.append("|---|---|---|---|---|---|")
        for mem in sorted(gate_ablation):
            on = gate_ablation[mem].get("on", {})
            off = gate_ablation[mem].get("off", {})
            delta = on.get("em", 0.0) - off.get("em", 0.0)
            lines.append(
                f"| {mem} | {on.get('store_size', 0):.0f} | {off.get('store_size', 0):.0f} | "
                f"{on.get('accepted', 0):.0f} | {off.get('accepted', 0):.0f} | {delta:+.3f} |"
            )
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
    gate_ablation: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
    plots: bool,
    seed_count: int,
    lineage: Dict[str, Dict[str, set]] | None = None,
) -> Dict[str, Path]:
    """Write per-suite reports and optional plots to ``out_dir``."""

    paths: Dict[str, Path] = {}
    for suite, presets in summary.items():
        suite_dir = out_dir / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        md_path = suite_dir / "summary.md"
        md_path.write_text(
            _render_markdown_suite(
                suite,
                presets,
                retrieval.get(suite),
                gates.get(suite),
                seed_count,
                lineage.get(suite) if lineage else None,
            )
        )
        paths[suite] = md_path
        if plots:
            _render_plots_suite(suite, presets, suite_dir)
    idx = _write_index(summary, paths, gates, gate_ablation, out_dir, seed_count)
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any suite lacks post_* metrics",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)
    date = args.date or _find_latest_date(runs_root)
    runs_path = runs_root / date
    if (runs_path / "INVALID").exists():
        log.warning("run %s marked invalid; skipping", runs_path)
        return
    metric_data = collect_metrics(runs_path)
    if args.strict:
        missing = _missing_post_metrics(metric_data)
        if missing:
            for suite, preset, size in missing:
                log.error(
                    "missing post_* metrics for suite=%s preset=%s size=%d",
                    suite,
                    preset,
                    size,
                )
            sys.exit(1)
    seed_count = max((len(v) for v in metric_data.values()), default=0)
    retrieval_data = collect_retrieval(runs_path)
    gate_data = collect_gates(runs_path)
    gate_ablation_data = collect_gate_ablation(runs_path)
    lineage_data = collect_lineage(runs_path)
    summary, missing_pre = summarise(metric_data)
    bad_pre = _missing_pre_suites(missing_pre)
    if bad_pre:
        for suite in bad_pre:
            miss, total = missing_pre[suite]
            frac = miss / total if total else 0.0
            log.error(
                "pre_* metrics missing for suite=%s (%.0f%% of runs)",
                suite,
                frac * 100,
            )
        sys.exit(1)
    retrieval_summary = summarise_retrieval(retrieval_data)
    gate_summary = summarise_gates(gate_data)
    out_dir = Path(args.out_dir) / date
    if args.smoke:
        write_smoke(Path(args.data_dir), out_dir / "smoke.md")
    paths = write_reports(
        summary,
        retrieval_summary,
        gate_summary,
        gate_ablation_data,
        out_dir,
        args.plots,
        seed_count,
        lineage_data,
    )
    for suite, md_path in paths.items():
        log.info("wrote %s for %s", md_path, suite)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
