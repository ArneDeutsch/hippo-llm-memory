#!/usr/bin/env python3
"""Shim to allow legacy `--mode`-style flags for eval_model.py."""
import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy CLI for eval_model.py")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    parser.add_argument("--mode", help="Phase: teach, replay, or test")
    parser.add_argument("--persist", help="Write to store_dir during teach/replay")
    parser.add_argument(
        "--store_dir", help="Base directory for persistent stores (e.g., runs/$DATE/stores)"
    )
    parser.add_argument(
        "--session_id",
        help="Logical session key; files are nested under store_dir/<algo>/<session_id>",
    )
    parser.add_argument(
        "--strict-telemetry",
        action="store_true",
        help="Fail fast on telemetry invariant violations",
    )
    parser.add_argument(
        "--pre-metrics",
        dest="pre_metrics",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Compute per-item metrics during the pre phase",
    )
    args, rest = parser.parse_known_args()

    overrides = list(args.overrides)
    if args.mode is not None:
        overrides.append(f"mode={args.mode}")
    if args.persist is not None:
        overrides.append(f"persist={args.persist}")
    store_dir = args.store_dir or os.getenv("STORES")
    if store_dir is not None:
        overrides.append(f"store_dir={store_dir}")
    session_id = args.session_id or os.getenv("HEI_SESSION_ID")
    if session_id is not None:
        overrides.append(f"session_id={session_id}")
    if args.strict_telemetry:
        overrides.append("strict_telemetry=true")
    if args.pre_metrics is not None:
        overrides.append(f"compute.pre_metrics={str(args.pre_metrics).lower()}")

    cmd = [sys.executable, "scripts/eval_model.py", *overrides, *rest]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
