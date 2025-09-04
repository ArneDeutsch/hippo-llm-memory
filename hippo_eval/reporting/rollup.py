"""Aggregation helpers for run reports."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, Tuple

from jinja2 import Environment, FileSystemLoader

from hippo_eval.reporting.health import Badge, render_panel
from hippo_eval.reporting.tables import display_name, make_summary_table
from hippo_mem.common.telemetry import validate_retrieval_snapshot

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

MetricDict = Dict[str, float]
MetricStat = tuple[float, float] | None
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


def collect_metrics(base: Path) -> Dict[Tuple[str, str, int], Iterable[MetricDict]]:
    """Collect metric dictionaries grouped by ``(suite, preset, size)``."""

    data: Dict[Tuple[str, str, int], list[MetricDict]] = defaultdict(list)
    for metrics_path in base.rglob("metrics.json"):
        parts = metrics_path.relative_to(base).parts
        if len(parts) < 3:
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


def missing_post_metrics(
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
    """Collect metrics for gate ON/OFF ablations."""

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
    """Collect lineage fields grouped by suite."""

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
    """Aggregate metrics with means and 95% CI bands."""

    summary: Summary = {}
    missing_pre: Dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for (suite, preset, size), metrics_list in data.items():
        if not metrics_list:
            continue
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


def add_longctx_uplift(summary: Summary) -> None:
    """Augment ``summary`` with uplift vs long-context baseline."""

    for suite, presets in summary.items():
        base = presets.get("baselines/longctx")
        if not base:
            continue
        for preset, sizes in presets.items():
            if preset == "baselines/longctx":
                continue
            for size, metrics in sizes.items():
                base_metrics = base.get(size)
                if not base_metrics:
                    continue
                for key in ("em", "em_norm", "f1"):
                    if (
                        key in metrics
                        and key in base_metrics
                        and metrics[key] is not None
                        and base_metrics[key] is not None
                    ):
                        uplift = metrics[key][0] - base_metrics[key][0]
                        metrics[f"uplift_vs_longctx_{key}"] = (uplift, 0.0)


def missing_pre_suites(
    missing_counts: Dict[str, tuple[int, int]], threshold: float = 0.2
) -> list[str]:
    """Return suites where ``pre_*`` metrics are missing beyond ``threshold``."""

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


def collect_runs(reports_root: Path):  # pragma: no cover - convenience
    """Return a DataFrame of run-level metrics under ``reports_root``."""

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("collect_runs requires pandas") from exc

    records: list[dict] = []
    for metrics_path in reports_root.rglob("metrics.json"):
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                record = json.load(f)
            records.append(record)
        except Exception:
            continue
    return pd.DataFrame(records)


def write_index(
    summary: Summary,
    suite_paths: Dict[str, Path],
    gates: Dict[str, Dict[str, Dict[str, MetricDict]]],
    gate_ablation: Dict[str, Dict[str, Dict[str, float]]],
    retrieval: Dict[str, Dict[str, MetricDict]],
    out_dir: Path,
    seed_count: int,
    lineage: Dict[str, Dict[str, set]] | None = None,
    missing_pre: Dict[str, tuple[int, int]] | None = None,
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
                em_by_preset[preset].append(agg[em_key] or 0.0)

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
    display = [display_name(k) for k in ordered]
    body: list[str] = []
    run_dir = Path("runs") / out_dir.name
    failed = list(run_dir.rglob("failed_preflight.json"))
    if failed:
        body.append("> ⚠️ preflight failures detected; see logs")
        body.append("")

    rows: list[dict[str, object]] = []
    for suite, preset, stats in rollup:
        row = {"suite": suite, "preset": preset}
        for key, disp in zip(ordered, display):
            val = stats.get(key)
            row[disp] = f"{val:.3f}" if isinstance(val, float) else "–"
        rows.append(row)
    table_md = make_summary_table(rows)
    body.append(table_md)
    body.append("")

    if retrieval:
        agg_ret: Dict[str, Dict[str, float]] = {}
        for suite_stats in retrieval.values():
            for mem, stats in suite_stats.items():
                dest = agg_ret.setdefault(mem, {k: 0.0 for k in stats})
                for k, v in stats.items():
                    dest[k] += float(v)
        tmpl = _ENV.get_template("partials/retrieval.md.j2")
        body.append(tmpl.render(retrieval=agg_ret).strip())
        body.append("")
    if gates:
        body.append("## Gate Telemetry")
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
            body.append(f"### Gate {status.upper()}")
            metric_keys = [k for k in order if any(k in s for s in mems.values())]
            header = "| mem | " + " | ".join(metric_keys) + " |"
            sep = "|---" * (len(metric_keys) + 1) + "|"
            body.extend([header, sep])
            for mem in sorted(mems):
                stats = mems[mem]
                vals: list[str] = []
                for key in metric_keys:
                    val = stats.get(key, 0)
                    if key in int_keys:
                        vals.append(str(int(val)))
                    else:
                        vals.append(f"{val:.3f}")
                body.append("| " + mem + " | " + " | ".join(vals) + " |")
            body.append("")
        if "on" in gates and "off" in gates:
            body.append("### Gate ON vs OFF")
            body.append("| mem | store_on | store_off | accepted_on | accepted_off | ΔEM |")
            body.append("|---|---|---|---|---|---|")
            on = gates["on"]
            off = gates["off"]

            def _metric(status: str, metric: str) -> float:
                for preset_name, size_stats in summary.get("episodic", {}).items():
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
                body.append(
                    f"| {mem} | {store_on:.0f} | {store_off:.0f} | {accepted_on:.0f} | {accepted_off:.0f} | {delta_em:+.3f} |"
                )
            body.append("")
    if gate_ablation:
        body.append("## Gate ON vs OFF")
        body.append("| mem | store_on | store_off | accepted_on | accepted_off | ΔEM |")
        body.append("|---|---|---|---|---|---|")
        for mem in sorted(gate_ablation):
            on = gate_ablation[mem].get("on", {})
            off = gate_ablation[mem].get("off", {})
            delta = on.get("em", 0.0) - off.get("em", 0.0)
            body.append(
                f"| {mem} | {on.get('store_size', 0):.0f} | {off.get('store_size', 0):.0f} | "
                f"{on.get('accepted', 0):.0f} | {off.get('accepted', 0):.0f} | {delta:+.3f} |"
            )
        body.append("")

    body.append("## Per-suite summaries")
    for suite in sorted(suite_paths):
        body.append(f"- [{suite}]({suite}/summary.md)")
    smoke = out_dir / "smoke.md"
    if smoke.exists():
        body.append("")
        body.append("See also: [smoke.md](smoke.md)")

    baseline_ok = True
    if missing_pre:
        baseline_ok = not missing_pre_suites(missing_pre)
    non_stub = True
    if lineage:
        for fields in lineage.values():
            srcs = fields.get("store_source")
            if srcs and "stub" in srcs:
                non_stub = False
                break
    gating_active = any(
        stats.get("attempts", 0) > 0
        for variants in gates.values()
        for mems in variants.values()
        for stats in mems.values()
    )
    retrieval_active = any(
        stats.get("requests", 0) > 0
        for suite_stats in retrieval.values()
        for stats in suite_stats.values()
    )
    badges = [
        Badge("BaselinesOK", baseline_ok, str(out_dir / "baselines")),
        Badge("NonStubStores", non_stub, str(out_dir / "stores")),
        Badge("GatingActive", gating_active, str(out_dir / "gates")),
        Badge("RetrievalActive", retrieval_active, str(out_dir / "retrieval")),
    ]
    tmpl = _ENV.get_template("index.md.j2")
    content = tmpl.render(health_panel=render_panel(badges), body="\n".join(body))
    index_path = out_dir / "index.md"
    index_path.write_text(content)
    return index_path


__all__ = [
    "MetricDict",
    "MetricStat",
    "MetricStats",
    "Summary",
    "load_metrics",
    "collect_metrics",
    "missing_post_metrics",
    "collect_retrieval",
    "collect_gates",
    "collect_gate_ablation",
    "collect_lineage",
    "summarise",
    "add_longctx_uplift",
    "missing_pre_suites",
    "summarise_retrieval",
    "summarise_gates",
    "collect_runs",
    "write_index",
]
