"""Validate expected persisted store layout before replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_eval.eval.store_utils import resolve_store_meta_path
from hippo_mem.utils.stores import (
    scan_episodic_store,
    scan_kg_store,
    validate_store,
)


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
    parser.add_argument(
        "--expect-nonzero-ratio",
        type=float,
        default=None,
        help="Minimum fraction of episodic keys with non-zero norm",
    )
    parser.add_argument(
        "--expect-nodes",
        type=int,
        default=None,
        help="Minimum node count for KG stores",
    )
    parser.add_argument(
        "--expect-edges",
        type=int,
        default=None,
        help="Minimum edge count for KG stores",
    )
    parser.add_argument(
        "--expect-embedding-coverage",
        type=float,
        default=None,
        help="Minimum embedding coverage for KG stores",
    )
    parser.add_argument(
        "--strict-telemetry",
        action="store_true",
        help="Fail fast on telemetry invariant violations",
    )
    return parser.parse_args()


def validate_cli_store(args: argparse.Namespace) -> Path | None:
    """Validate store layout and content for a run."""

    run_id = args.run_id or os.environ.get("RUN_ID")
    if not run_id:
        print("RUN_ID is required; set RUN_ID env or pass --run_id", file=sys.stderr)
        raise SystemExit(1)

    try:
        preset = args.preset or f"memory/{args.algo}"
        path = validate_store(
            run_id=run_id,
            preset=preset,
            algo=args.algo,
            kind=args.kind,
        )
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


def _find_metrics(run_id: str, algo: str) -> Path | None:
    """Return the first metrics.json for ``run_id``/``algo`` if present."""

    base = Path("runs") / run_id / algo
    for p in base.rglob("metrics.json"):
        return p
    return None


def _scan_spatial(path: Path) -> tuple[int, int]:
    """Return node and edge counts for spatial stores."""

    nodes = edges = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = rec.get("type")
            if typ == "node":
                nodes += 1
            elif typ == "edge":
                edges += 1
    return nodes, edges


def strict_telemetry_checks(
    path: Path | None,
    kind: str,
    run_id: str,
    algo: str,
    metrics_path: str | None,
    nonzero_ratio: float | None = None,
    expect_nodes: int | None = None,
    expect_edges: int | None = None,
) -> None:
    """Enforce size, embedding, and audit invariants."""

    mp = Path(metrics_path) if metrics_path else _find_metrics(run_id, algo)
    if mp is None:
        raise FileNotFoundError("metrics.json required for strict telemetry")
    with mp.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    n = int(metrics.get("n", 0))
    if path is not None:
        if kind == "episodic":
            count, nz = scan_episodic_store(path)
            ratio = nz / count if count else 0.0
            expected = max(int(n * 0.8), 1 if n else 0)
            if count < expected:
                raise ValueError(f"episodic traces {count} < expected {expected}")
            threshold = 0.9 if nonzero_ratio is None else nonzero_ratio
            if ratio < threshold:
                raise ValueError(f"episodic non-zero ratio {ratio:.2f} < {threshold}")
        elif kind in {"kg", "relational"}:
            nodes, edges, node_nz, edge_nz = scan_kg_store(path)
            expected_nodes = (
                expect_nodes if expect_nodes is not None else max(int(n * 2), 1 if n else 0)
            )
            expected_edges = (
                expect_edges if expect_edges is not None else max(int(n * 2), 1 if n else 0)
            )
            if nodes < expected_nodes or edges < expected_edges:
                raise ValueError(
                    "relational store too small: "
                    f"nodes {nodes} < {expected_nodes} or edges {edges} < {expected_edges}"
                )
            nr = node_nz / nodes if nodes else 0.0
            er = edge_nz / edges if edges else 0.0
            if nr < 0.9 or er < 0.9:
                raise ValueError("relational embedding ratio < 0.9")
        elif kind in {"spatial", "map"}:
            nodes, edges = _scan_spatial(path)
            expected = max(int(n * 0.8), 1 if n else 0)
            if nodes < expected or edges < expected:
                raise ValueError(
                    f"spatial store too small: nodes {nodes} edges {edges} < {expected}"
                )

    audit = mp.with_name("audit_sample.jsonl")
    retrieval = metrics.get("retrieval", {})
    reqs = sum(r.get("requests", 0) for r in retrieval.values())
    if reqs > 0:
        if not audit.exists():
            raise FileNotFoundError(f"missing audit sample: {audit}")
        missing: list[int] = []
        with audit.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                router = rec.get("router_path") or []
                if router and "injected_context" not in rec:
                    missing.append(rec.get("id", -1))
                    if len(missing) >= 5:
                        break
        if missing:
            raise ValueError(
                "audit rows missing injected_context: " + ", ".join(str(m) for m in missing)
            )


def threshold_checks(path: Path | None, kind: str, args: argparse.Namespace) -> None:
    """Apply user-specified expectations on the store."""

    if path is None:
        return
    if kind == "episodic" and args.expect_nonzero_ratio is not None:
        count, nz = scan_episodic_store(path)
        ratio = nz / count if count else 0.0
        if ratio < args.expect_nonzero_ratio:
            raise ValueError(f"episodic non-zero ratio {ratio:.2f} < {args.expect_nonzero_ratio}")
    elif kind in {"kg", "relational"}:
        nodes, edges, node_nz, edge_nz = scan_kg_store(path)
        if args.expect_nodes is not None and nodes < args.expect_nodes:
            raise ValueError(f"kg nodes {nodes} < expected {args.expect_nodes}")
        if args.expect_edges is not None and edges < args.expect_edges:
            raise ValueError(f"kg edges {edges} < expected {args.expect_edges}")
        if args.expect_embedding_coverage is not None:
            total = nodes + edges
            coverage = (node_nz + edge_nz) / total if total else 0.0
            if coverage < args.expect_embedding_coverage:
                raise ValueError(
                    "kg embedding coverage " f"{coverage:.2f} < {args.expect_embedding_coverage}"
                )


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    run_id = args.run_id or os.environ.get("RUN_ID") or ""
    path = validate_cli_store(args)
    try:
        verify_metrics(path, args.kind, args.metrics)
        if args.strict_telemetry:
            strict_telemetry_checks(
                path,
                args.kind,
                run_id,
                args.algo,
                args.metrics,
                args.expect_nonzero_ratio,
                args.expect_nodes,
                args.expect_edges,
            )
        threshold_checks(path, args.kind, args)
    except (FileNotFoundError, ValueError) as err:
        print(err, file=sys.stderr)
        raise SystemExit(1) from err
    if path is None:
        print(f"OK: no store for baseline {args.preset}")
    else:
        print(f"OK: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
