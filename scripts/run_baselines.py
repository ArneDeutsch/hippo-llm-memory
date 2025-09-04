#!/usr/bin/env python3
"""Aggregate baseline metrics and persist summary with confidence intervals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from hippo_eval.baselines import aggregate_metrics
from hippo_eval.harness.io import write_metrics
from hippo_mem.utils import validate_run_id


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Root runs directory")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_id = validate_run_id(args.run_id)

    root = Path(args.runs_dir) / run_id / "baselines"
    rows = aggregate_metrics(root)
    write_metrics(rows, root)
    print(f"aggregated {len(rows)} baseline rows under {root}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
