#!/usr/bin/env python3
"""Shim to allow legacy `--mode`-style flags for eval_model.py."""
import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Legacy CLI for eval_model.py",
        epilog=(
            "Run baselines first: python -m hippo_eval.baselines --run-id <RID>\n"
            "store_dir: runs/<RID>/stores (recommended) or runs/<RID>/stores/<algo>.\n"
            "Replay cycles: pass replay_cycles=N or replay.cycles=N."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    parser.add_argument("--mode", help="Phase: teach, replay, or test")
    parser.add_argument(
        "--persist",
        dest="persist",
        action=argparse.BooleanOptionalAction,
        help="Write to store_dir during teach/replay",
    )
    parser.add_argument(
        "--store_dir", help="Base directory for persistent stores (e.g., runs/$RUN_ID/stores)"
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
    parser.add_argument("--run-id", help="Run identifier")
    parser.add_argument("--verbose", action="store_true", help="Print resolved run_id")
    parser.add_argument(
        "--isolate",
        choices=["per_item", "per_episode", "none"],
        help="Isolation level for memory stores",
    )
    args, rest = parser.parse_known_args()

    overrides = list(args.overrides)

    def _has(key: str) -> bool:
        prefix = f"{key}="
        return any(o.startswith(prefix) for o in overrides)

    if args.mode is not None:
        overrides.append(f"mode={args.mode}")
    if args.persist is not None:
        overrides.append(f"persist={args.persist}")

    if args.store_dir is not None:
        overrides.append(f"store_dir={args.store_dir}")
    elif not _has("store_dir"):
        store_dir = os.getenv("STORES")
        if store_dir is not None:
            overrides.append(f"store_dir={store_dir}")

    if args.session_id is not None:
        overrides.append(f"session_id={args.session_id}")
    elif not _has("session_id"):
        session_id = os.getenv("HEI_SESSION_ID")
        if session_id is not None:
            overrides.append(f"session_id={session_id}")

    if args.strict_telemetry:
        overrides.append("strict_telemetry=true")
    if args.pre_metrics is not None:
        overrides.append(f"compute.pre_metrics={str(args.pre_metrics).lower()}")

    if args.run_id is not None:
        run_id = args.run_id
    elif not _has("run_id"):
        run_id = os.getenv("RUN_ID")
    else:
        run_id = None
    if run_id is not None:
        overrides.append(f"run_id={run_id}")
        if args.verbose:
            print(f"run_id={run_id}", file=sys.stderr)

    if args.isolate is not None:
        overrides.append(f"isolate={args.isolate}")

    cmd = [sys.executable, "scripts/eval_model.py", *overrides, *rest]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
