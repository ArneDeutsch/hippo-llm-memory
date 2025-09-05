"""Validate expected persisted store layout before replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_eval.eval.store_utils import resolve_store_meta_path
from hippo_mem.utils.stores import validate_store


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", help="Run identifier; defaults to $RUN_ID env var")
    parser.add_argument("--algo", default="hei_nw", help="Memory algorithm identifier")
    parser.add_argument("--kind", default="episodic", help="Store kind to validate")
    parser.add_argument(
        "--preset", help="Preset identifier; baselines must not produce stores", default=None
    )
    parser.add_argument("--metrics", help="Path to metrics.json for count validation", default=None)
    return parser.parse_args()


def validate_cli_store(args: argparse.Namespace) -> Path | None:
    """Validate store layout and content for a run."""

    run_id = args.run_id or os.environ.get("RUN_ID")
    if not run_id:
        print("RUN_ID is required; set RUN_ID env or pass --run_id", file=sys.stderr)
        raise SystemExit(1)

    try:
        path = validate_store(run_id=run_id, algo=args.algo, kind=args.kind, preset=args.preset)
    except (FileExistsError, FileNotFoundError, ValueError) as err:  # pragma: no cover - CLI
        print(err, file=sys.stderr)
        raise SystemExit(1) from err

    if path is None:
        return None

    session_id = path.parent.name
    store_dir = path.parents[2]
    preset = args.preset or f"memory/{args.algo}"
    meta = resolve_store_meta_path(preset, store_dir, session_id)
    if not meta.exists():
        raise FileNotFoundError(f"missing store_meta.json: {meta}")
    has_data = False
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                has_data = True
                break
    if not has_data:
        raise ValueError(
            "empty store: "
            f"{path} — run:\n  python scripts/eval_model.py --mode teach --run-id {run_id}\n"
            "hint: teach path must persist data.\n"
            "  - SGC-RSS: ensure tuples are written (gate accepts and schemas are seeded or direct upsert is used).\n"
            "  - SMPD: ensure the spatial map writes nodes/edges (no blank JSONL on teach-only)."
        )
    return path


def _expected_lines(kind: str, per_mem: dict, diag: dict) -> int:
    """Return expected line count based on metrics and store kind."""

    key_map = {"episodic": "episodic", "kg": "relational", "map": "spatial", "spatial": "spatial"}
    strategies = {
        "kg": lambda key, pm, dg: int(pm.get(key, 0))
        + int(dg.get("relational", {}).get("nodes_added", 0)),
        "map": lambda key, pm, dg: int(pm.get(key, 0)) + 1,
        "spatial": lambda key, pm, dg: int(pm.get(key, 0)) + 1,
    }
    key = key_map.get(kind, kind)
    strategy = strategies.get(kind, lambda k, pm, dg: int(pm.get(k, 0)))
    return strategy(key, per_mem, diag)


def verify_metrics(path: Path | None, kind: str, metrics_path: str | None) -> None:
    """Verify metrics line counts against store contents."""

    if not metrics_path:
        return

    with Path(metrics_path).open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    store = metrics.get("store", {})
    per_mem = store.get("per_memory", {})
    diag = store.get("diagnostics", {})
    expected = _expected_lines(kind, per_mem, diag)

    if path is None:
        if expected != 0:
            raise ValueError("metrics report store entries but no file found")
        return

    actual = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                actual += 1
                continue
            val = rec.get("value")
            if isinstance(val, dict) and val.get("provenance") == "dummy":
                continue
            actual += 1
    if actual != expected:
        raise ValueError(f"{kind} store line count {actual} != metrics expectation {expected}")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    path = validate_cli_store(args)
    verify_metrics(path, args.kind, args.metrics)
    if path is None:
        print(f"OK: no store for baseline {args.preset}")
    else:
        print(f"OK: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
