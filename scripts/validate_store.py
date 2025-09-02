"""Validate expected persisted store layout before replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_mem.eval.store_utils import resolve_store_meta_path
from hippo_mem.utils.stores import validate_store


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", help="Run identifier; defaults to $RUN_ID or $DATE env vars")
    parser.add_argument("--algo", default="hei_nw", help="Memory algorithm identifier")
    parser.add_argument("--kind", default="episodic", help="Store kind to validate")
    parser.add_argument(
        "--preset", help="Preset identifier; baselines must not produce stores", default=None
    )
    parser.add_argument("--metrics", help="Path to metrics.json for count validation", default=None)
    args = parser.parse_args()

    try:
        path = validate_store(
            run_id=args.run_id, algo=args.algo, kind=args.kind, preset=args.preset
        )
    except (FileExistsError, FileNotFoundError) as err:  # pragma: no cover - tested via CLI
        print(err, file=sys.stderr)
        raise SystemExit(1) from err
    if path is not None:
        session_id = path.parent.name
        store_dir = path.parents[2]
        preset = args.preset or f"memory/{args.algo}"
        meta = resolve_store_meta_path(preset, store_dir, session_id)
        if not meta.exists():
            raise FileNotFoundError(f"missing store_meta.json: {meta}")

    if args.metrics:
        with Path(args.metrics).open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        store = metrics.get("store", {})
        per_mem = store.get("per_memory", {})
        diag = store.get("diagnostics", {})
        key_map = {
            "episodic": "episodic",
            "kg": "relational",
            "map": "spatial",
            "spatial": "spatial",
        }
        key = key_map.get(args.kind, args.kind)
        expected = int(per_mem.get(key, 0))
        if args.kind == "kg":
            expected += int(diag.get("relational", {}).get("nodes_added", 0))
        elif args.kind in ("map", "spatial"):
            expected += 1  # meta line
        if path is None:
            if expected != 0:
                raise ValueError("metrics report store entries but no file found")
        else:
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
                raise ValueError(
                    f"{args.kind} store line count {actual} != metrics expectation {expected}"
                )
    if path is None:
        print(f"OK: no store for baseline {args.preset}")
    else:
        print(f"OK: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
