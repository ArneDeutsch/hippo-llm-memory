# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Generate Markdown reports from evaluation runs."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from hippo_eval.reporting import plots as plot_utils
from hippo_eval.reporting import rollup, tables

log = logging.getLogger(__name__)

SLUG_RE = re.compile(r"^[A-Za-z0-9._-]{3,64}$")


def _date_str(value: object | None) -> str:
    """Return a normalized date string."""

    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    date = str(value)
    if "_" not in date and date.isdigit() and len(date) > 8:
        return f"{date[:8]}_{date[8:]}"
    return date


def _load_suite_metrics(base: Path) -> Dict[str, dict]:
    """Return metrics.json records keyed by suite."""

    data: Dict[str, dict] = {}
    if not base.exists():
        return data
    for metrics_path in base.rglob("metrics.json"):
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                record = json.load(f)
        except json.JSONDecodeError:
            continue
        suite = record.get("suite") or metrics_path.parent.parent.name
        data[suite] = record
    return data


def _suite_metric(records: Dict[str, dict], suite: str, key: str) -> float:
    """Extract a metric value for ``suite`` from loaded ``records``."""

    return float(records.get(suite, {}).get("metrics", {}).get(suite, {}).get(key, 0.0))


def _missing_post_metrics(
    metric_data: Dict[tuple[str, str, int], Iterable[rollup.MetricDict]],
) -> list[tuple[str, str, int]]:
    """Return groups lacking any ``post_*`` metrics (compat shim)."""

    return rollup.missing_post_metrics(metric_data)


def sanity_sweep(runs_dir: Path, run_id: str) -> None:
    """Print store sizes, gate stats, hit rates and EM/F1 deltas for ``run_id``."""

    run_path = runs_dir / run_id
    core = _load_suite_metrics(run_path / "baselines" / "core")
    longctx = _load_suite_metrics(run_path / "baselines" / "longctx")
    memory = _load_suite_metrics(run_path / "memory")
    kind_map = {"episodic": "episodic", "semantic": "relational", "spatial": "spatial"}
    header = (
        f"{'suite':12} {'store':>5} {'gate':>7} {'hit':>5} "
        f"{'EM(core/long/mem)':>23} {'F1(core/long/mem)':>23}"
    )
    print(header)
    flags: list[str] = []
    for suite, record in sorted(memory.items()):
        kind = kind_map.get(suite, suite)
        store = int(record.get("store", {}).get("size", 0))
        gate = record.get("gating", {}).get(kind, {})
        attempts = int(gate.get("attempts", 0))
        accepted = int(gate.get("accepted", 0))
        hit = float(record.get("retrieval", {}).get(kind, {}).get("hit_rate_at_k", 0.0))
        em = _suite_metric(memory, suite, "pre_em")
        f1 = _suite_metric(memory, suite, "pre_f1")
        core_em = _suite_metric(core, suite, "pre_em")
        long_em = _suite_metric(longctx, suite, "pre_em")
        core_f1 = _suite_metric(core, suite, "pre_f1")
        long_f1 = _suite_metric(longctx, suite, "pre_f1")
        print(
            f"{suite:12} {store:5d} {attempts:3d}/{accepted:<3d} {hit:5.2f} "
            f"{core_em:5.2f}/{long_em:5.2f}/{em:5.2f} "
            f"{core_f1:5.2f}/{long_f1:5.2f}/{f1:5.2f}"
        )
        if store == 0 or attempts == 0 or hit == 0.0:
            flags.append(f"{suite} memory inert")
        if long_em >= 0.98:
            flags.append(f"{suite} baseline saturated (longctx EM {long_em:.2f})")
    if flags:
        print("\nRed flags:")
        for msg in flags:
            print(f"- {msg}")


def _find_latest_run_id(root: Path) -> str:
    """Return the latest run identifier available under ``root``."""

    ids = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not ids:
        msg = f"no run directories found under {root}"
        raise FileNotFoundError(msg)
    return ids[-1]


def write_smoke(data_root: Path, out_path: Path, n_rows: int = 3) -> Path:
    """Write a sample of dataset rows for sanity checking."""

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


def write_reports(
    summary: rollup.Summary,
    retrieval: Dict[str, Dict[str, rollup.MetricDict]],
    gates: Dict[str, Dict[str, Dict[str, rollup.MetricDict]]],
    gate_ablation: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
    plots: bool,
    seed_count: int,
    lineage: Dict[str, Dict[str, set]] | None = None,
    missing_pre: Dict[str, tuple[int, int]] | None = None,
) -> Dict[str, Path]:
    """Write per-suite reports and optional plots to ``out_dir``."""

    if missing_pre is None:
        counts: Dict[str, tuple[int, int]] = {}
        for suite, presets in summary.items():
            miss = 0
            total = 0
            for preset_stats in presets.values():
                for metrics in preset_stats.values():
                    for key, stat in metrics.items():
                        if key.startswith("pre_"):
                            total += 1
                            if stat is None:
                                miss += 1
            counts[suite] = (miss, total)
        missing_pre = counts

    paths: Dict[str, Path] = {}
    for suite, presets in summary.items():
        suite_dir = out_dir / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        md_path = suite_dir / "summary.md"
        md_path.write_text(
            tables.render_markdown_suite(
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
            plot_utils.render_suite_plots(suite, presets, suite_dir, retrieval.get(suite))
    idx = rollup.write_index(
        summary,
        paths,
        gates,
        gate_ablation,
        retrieval,
        out_dir,
        seed_count,
        lineage,
        missing_pre,
    )
    log.info("wrote %s", idx)
    return paths


def render_run_report(
    run_dir: Path,
    out_dir: Path,
    plots: bool = False,
    data_dir: Path | None = None,
    smoke: bool = False,
    strict: bool = False,
) -> None:
    """Render a full report for ``run_dir`` into ``out_dir``."""

    metric_data = rollup.collect_metrics(run_dir)
    if strict:
        missing = rollup.missing_post_metrics(metric_data)
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
    retrieval_data = rollup.collect_retrieval(run_dir)
    gate_data = rollup.collect_gates(run_dir)
    gate_ablation_data = rollup.collect_gate_ablation(run_dir)
    lineage_data = rollup.collect_lineage(run_dir)
    summary, missing_pre = rollup.summarise(metric_data)
    rollup.add_longctx_uplift(summary)
    bad_pre = rollup.missing_pre_suites(missing_pre)
    if bad_pre and strict:
        for suite in bad_pre:
            miss, total = missing_pre[suite]
            frac = miss / total if total else 0.0
            log.error(
                "pre_* metrics missing for suite=%s (%.0f%% of runs)",
                suite,
                frac * 100,
            )
        sys.exit(1)
    retrieval_summary = rollup.summarise_retrieval(retrieval_data)
    gate_summary = rollup.summarise_gates(gate_data)
    if smoke and data_dir is not None:
        write_smoke(data_dir, out_dir / "smoke.md")
    write_reports(
        summary,
        retrieval_summary,
        gate_summary,
        gate_ablation_data,
        out_dir,
        plots,
        seed_count,
        lineage_data,
        missing_pre,
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="directory containing run outputs")
    parser.add_argument("--out-dir", default="reports", help="directory to write reports to")
    parser.add_argument("--data-dir", default="data", help="dataset directory for smoke report")
    parser.add_argument("--run-id", dest="run_id", default=None, help="run identifier")
    parser.add_argument("--date", dest="date", default=None, help="deprecated run date")
    parser.add_argument("--plots", action="store_true", help="render bar plots using matplotlib")
    parser.add_argument("--smoke", action="store_true", help="also write smoke.md with sample rows")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any suite lacks post_* metrics",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)
    run_id = args.run_id
    if not run_id and args.date:
        run_id = _date_str(args.date)
        log.warning("`--date` is deprecated; use --run-id")
    if not run_id:
        run_id = _find_latest_run_id(runs_root)
    if not SLUG_RE.match(run_id):
        raise ValueError("run_id must match ^[A-Za-z0-9._-]{3,64}$")
    runs_path = runs_root / run_id
    if (runs_path / "INVALID").exists():
        log.warning("run %s marked invalid; skipping", runs_path)
        return
    out_dir = Path(args.out_dir) / run_id
    render_run_report(
        runs_path,
        out_dir,
        plots=args.plots,
        data_dir=Path(args.data_dir),
        smoke=args.smoke,
        strict=args.strict,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
