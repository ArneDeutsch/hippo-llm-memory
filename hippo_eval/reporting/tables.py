# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Helpers for assembling report tables."""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from typing import Dict

from hippo_eval.reporting.table_utils import (
    ENV,
    display_name,
    format_stat,
    make_metrics_table,
)


class SuiteOrdering(Enum):
    """Row ordering strategies for different suites."""

    DEFAULT = "default"
    SEMANTIC = "semantic"

    @staticmethod
    def for_suite(suite: str) -> "SuiteOrdering":
        return SuiteOrdering.SEMANTIC if suite == "semantic" else SuiteOrdering.DEFAULT

    def order(
        self, presets: Dict[str, Dict[int, Dict[str, tuple[float, float]]]]
    ) -> list[tuple[str, int, Dict[str, tuple[float, float]]]]:
        if self is SuiteOrdering.SEMANTIC:
            rows: list[tuple[float, str, int, Dict[str, tuple[float, float]]]] = []
            for preset, sizes in presets.items():
                for size, metrics in sizes.items():
                    em_raw_stat = metrics.get("em_raw") or (0.0, 0.0)
                    rows.append((em_raw_stat[0], preset, size, metrics))
            rows.sort(key=lambda x: x[0], reverse=True)
            return [(preset, size, metrics) for _, preset, size, metrics in rows]
        rows: list[tuple[str, int, Dict[str, tuple[float, float]]]] = []
        for preset in sorted(presets):
            for size in sorted(presets[preset]):
                rows.append((preset, size, presets[preset][size]))
        return rows


def build_lineage_section(lineage: Dict[str, set] | None) -> list[str]:
    """Return lineage pill lines."""

    if not lineage:
        return []
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
        return ["> " + " ".join(pills), ""]
    return []


def metric_ordering(
    presets: Dict[str, Dict[int, Dict[str, tuple[float, float]]]],
) -> tuple[list[str], list[str]]:
    """Return ordered metric keys and display names."""

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
    display = [display_name(k) for k in ordered]
    return ordered, display


def summarize_presets(
    suite: str,
    presets: Dict[str, Dict[int, Dict[str, tuple[float, float]]]],
    ordered: list[str],
    display: list[str],
    seed_count: int,
) -> list[str]:
    """Return lines summarizing per-preset metrics."""

    lines: list[str] = []
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
        lines.extend(["> ⚠️ non-informative for uplift: pre_em_norm ≥ 0.98", ""])

    lines.append("## Uplift")
    if seed_count <= 1:
        lines.extend(["> single-seed run: CI bands unavailable", ""])

    ordering = SuiteOrdering.for_suite(suite)
    rows_sorted = ordering.order(presets)

    baseline_pre = 0.0
    for preset, sizes in presets.items():
        if preset.startswith("baselines/"):
            first = next(iter(sizes.values()))
            baseline_pre = (first.get("pre_em_norm") or first.get("em_norm") or (0.0, 0.0))[0]
            break

    warnings_summary: list[str] = []
    table_rows: list[OrderedDict[str, str | int]] = []
    for preset, size, metrics in rows_sorted:
        row: OrderedDict[str, str | int] = OrderedDict()
        row["Preset"] = preset
        row["Size"] = size
        warn: list[str] = []
        for key, disp in zip(ordered, display):
            stat = metrics.get(key)
            if stat is None:
                if key.startswith("pre_"):
                    row[disp] = "_missing_"
                    if "MissingPre" not in warn:
                        warn.append("MissingPre")
                else:
                    row[disp] = "–"
            else:
                row[disp] = format_stat(stat)
        note = ""
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
        if preset.startswith("memory/") and all(r == 0 for r in retrieval_reqs):
            warn.append("NoRetrieval")
        if warn:
            note = "⚠️ " + ", ".join(warn)
            warnings_summary.append(f"{preset} {size}: {', '.join(warn)}")
        row["⚠️"] = note
        table_rows.append(row)

    table_md = make_metrics_table(table_rows)
    lines.append(table_md)
    lines.append("")
    if warnings_summary:
        lines.append("### Warnings")
        for w in warnings_summary:
            lines.append(f"- {w}")
        lines.append("")
    return lines


def render_retrieval_section(retrieval: Dict[str, Dict[str, float]] | None) -> list[str]:
    """Return retrieval section lines."""

    if not retrieval:
        return []
    tmpl = ENV.get_template("partials/retrieval.md.j2")
    return [tmpl.render(retrieval=retrieval).strip(), ""]


def render_gate_telemetry(
    gates: Dict[str, Dict[str, Dict[str, float]]] | None,
    presets: Dict[str, Dict[int, Dict[str, tuple[float, float]]]],
) -> list[str]:
    """Return gate telemetry lines."""

    if not gates:
        return []
    lines: list[str] = ["## Gate Telemetry"]
    int_keys = {
        "attempts",
        "accepted",
        "blocked",
        "skipped",
        "inserted",
        "aggregated",
        "routed_to_episodic",
        "blocked_new_edges",
        "nodes_added",
        "edges_added",
    }
    order = [
        "attempts",
        "accepted",
        "blocked",
        "skipped",
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
                f"| {mem} | {store_on:.0f} | {store_off:.0f} | {accepted_on:.0f} | "
                f"{accepted_off:.0f} | {delta_em:+.3f} |"
            )
        lines.append("")
    return lines


def render_markdown_suite(
    suite: str,
    presets: Dict[str, Dict[int, Dict[str, tuple[float, float]]]],
    retrieval: Dict[str, Dict[str, float]] | None,
    gates: Dict[str, Dict[str, Dict[str, float]]] | None,
    seed_count: int,
    lineage: Dict[str, set] | None = None,
) -> str:
    """Return a Markdown table for a single suite."""

    lines: list[str] = [f"# {suite} Summary", ""]
    lines.extend(build_lineage_section(lineage))
    if presets:
        ordered, display = metric_ordering(presets)
        lines.extend(summarize_presets(suite, presets, ordered, display, seed_count))
    lines.extend(render_retrieval_section(retrieval))
    lines.extend(render_gate_telemetry(gates, presets))
    return "\n".join(lines)


__all__ = ["render_markdown_suite"]
