"""Helpers for assembling report tables."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader

_DISPLAY_NAMES = {
    "em_raw": "EM (raw)",
    "em_norm": (
        '<span title="Normalized exact match (lowercase, no punctuation or '
        'articles)">EM (norm)</span>'
    ),
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
    "memory_hit_rate": "memory_hit_rate",
    "latency_ms_delta": "latency_ms_delta",
    "uplift_vs_longctx_em": "ΔEM vs longctx",
    "uplift_vs_longctx_f1": "ΔF1 vs longctx",
}

TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def display_name(key: str) -> str:
    """Return a human-friendly display name for ``key``."""

    return _DISPLAY_NAMES.get(key, key)


def format_stat(stat: tuple[float, float]) -> str:
    """Format a ``(mean, ci)`` tuple with 3 decimals."""

    mean_val, ci = stat
    return f"{mean_val:.3f} ± {ci:.3f}"


def _normalize(df: object) -> tuple[list[Mapping[str, object]], Sequence[str]]:
    """Return records and columns for ``df`` supporting DataFrame-like objects."""

    if hasattr(df, "to_dict") and hasattr(df, "columns"):
        records = df.to_dict(orient="records")  # type: ignore[arg-type]
        columns = list(df.columns)  # type: ignore[attr-defined]
        return records, columns
    if isinstance(df, Iterable):
        records = [dict(r) for r in df]  # type: ignore[misc]
        columns = list(records[0].keys()) if records else []
        return records, columns
    raise TypeError("df must be a DataFrame or iterable of mappings")


def _make_table(records: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for rec in records:
        row = [str(rec.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_metrics_table(df: object) -> str:
    """Return a Markdown metrics table for DataFrame ``df``."""

    records, columns = _normalize(df)
    return _make_table(records, columns)


def make_summary_table(df: object) -> str:
    """Return a Markdown summary table for DataFrame ``df``."""

    records, columns = _normalize(df)
    return _make_table(records, columns)


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
            lines.extend(["> ⚠️ non-informative for uplift: pre_em_norm ≥ 0.98", ""])

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
        lines.append("## Uplift")
        if seed_count <= 1:
            lines.extend(["> single-seed run: CI bands unavailable", ""])
        rows_sorted: list[tuple[float, str, int, Dict[str, tuple[float, float]]]] = []
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
                    rows_sorted.append((em_raw_stat[0], preset, size, metrics))
            rows_sorted.sort(key=lambda x: x[0], reverse=True)
        else:
            for preset in sorted(presets):
                sizes = presets[preset]
                for size in sorted(sizes):
                    rows_sorted.append((0.0, preset, size, sizes[size]))
        warnings_summary: list[str] = []
        table_rows: list[OrderedDict[str, str | int]] = []
        for _, preset, size, metrics in rows_sorted:
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
    if retrieval:
        tmpl = _ENV.get_template("partials/retrieval.md.j2")
        lines.append(tmpl.render(retrieval=retrieval).strip())
        lines.append("")
    if gates:
        lines.append("## Gate Telemetry")
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
                    f"| {mem} | {store_on:.0f} | {store_off:.0f} | {accepted_on:.0f} | {accepted_off:.0f} | {delta_em:+.3f} |"
                )
            lines.append("")
    return "\n".join(lines)


__all__ = [
    "TEMPLATE_DIR",
    "_ENV",
    "display_name",
    "format_stat",
    "make_metrics_table",
    "make_summary_table",
    "render_markdown_suite",
]
