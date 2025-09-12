# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
#!/usr/bin/env python3
"""Summarize parameter sweep runs and prune non-informative cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

BaselineMap = Dict[str, Tuple[float, float]]


def load_baselines(root: Path) -> BaselineMap:
    """Return mapping ``suite -> (em_raw_mean, f1_mean)`` from ``metrics.csv``.

    Parameters
    ----------
    root:
        Directory ``runs/<date>/baselines`` containing ``metrics.csv``.
    """

    path = root / "metrics.csv"
    if not path.exists():  # pre: metrics must exist
        msg = f"baseline metrics not found under {root}"
        raise FileNotFoundError(msg)

    baselines: BaselineMap = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            suite = row["suite"]
            baselines[suite] = (
                float(row["em_raw_mean"]),
                float(row["f1_mean"]),
            )
    if not baselines:  # post: expect at least one baseline
        msg = f"no baseline rows found in {path}"
        raise ValueError(msg)
    return baselines


def _flatten_config(cfg: Dict[str, Any], prefix: str = "") -> List[str]:
    """Flatten nested ``cfg`` to dotted ``key=value`` strings."""

    items: List[str] = []
    for key, val in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            items.extend(_flatten_config(val, full_key))
        else:
            items.append(f"{full_key}={val}")
    return items


def _is_informative(delta_em: float, delta_f1: float, eps: float) -> bool:
    """Return ``True`` if any delta exceeds ``eps``."""

    return abs(delta_em) >= eps or abs(delta_f1) >= eps


def process_sweep_dir(path: Path, baselines: BaselineMap, eps: float) -> Dict[str, Any]:
    """Analyze a single sweep directory and mark non-informative ones.

    Parameters
    ----------
    path:
        Directory containing ``metrics.json`` and ``meta.json``.
    baselines:
        Mapping from suite to baseline metrics.
    eps:
        Minimum delta required to retain a sweep.
    """

    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))

    suite = meta.get("suite")
    if suite not in baselines:  # pre: baseline must exist for suite
        msg = f"missing baseline metrics for suite '{suite}'"
        raise KeyError(msg)

    em_raw = float(metrics["metrics"][suite]["pre_em_raw"])
    f1 = float(metrics["metrics"][suite]["pre_f1"])

    base_em_raw, base_f1 = baselines[suite]
    delta_em = em_raw - base_em_raw
    delta_f1 = f1 - base_f1
    informative = _is_informative(delta_em, delta_f1, eps)

    if not informative:
        (path / "non_informative.flag").write_text("flat\n", encoding="utf-8")

    config_hint = ", ".join(_flatten_config(meta.get("config", {})))

    return {
        "name": path.name,
        "suite": suite,
        "em_raw": em_raw,
        "f1": f1,
        "delta_em_raw": delta_em,
        "delta_f1": delta_f1,
        "informative": informative,
        "config_hint": config_hint,
    }


def summarize_sweeps(sweeps_dir: Path, baselines: BaselineMap, eps: float) -> Dict[str, Any]:
    """Process all sweeps under ``sweeps_dir`` and write summary files."""

    rows: List[Dict[str, Any]] = []
    for child in sweeps_dir.iterdir():
        if child.is_dir():
            rows.append(process_sweep_dir(child, baselines, eps))

    pruned = sum(1 for r in rows if not r["informative"])
    summary = {"epsilon": eps, "rows": rows, "pruned": pruned}

    csv_path = sweeps_dir / "summary.csv"
    fieldnames = [
        "name",
        "suite",
        "em_raw",
        "f1",
        "delta_em_raw",
        "delta_f1",
        "informative",
        "config_hint",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = sweeps_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Root runs directory")
    parser.add_argument("--date", required=True, help="Run identifier date")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Minimum delta to retain sweep")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.runs_dir) / args.date
    baselines = load_baselines(root / "baselines")
    summarize_sweeps(root / "sweeps", baselines, args.epsilon)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
