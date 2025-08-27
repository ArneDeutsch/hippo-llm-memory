#!/usr/bin/env python3
"""Shim to allow legacy `--mode`-style flags for eval_model.py."""
import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy CLI for eval_model.py")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    parser.add_argument("--mode")
    parser.add_argument("--persist")
    parser.add_argument("--store_dir")
    parser.add_argument("--session_id")
    args, rest = parser.parse_known_args()

    overrides = list(args.overrides)
    if args.mode is not None:
        overrides.append(f"mode={args.mode}")
    if args.persist is not None:
        overrides.append(f"persist={args.persist}")
    if args.store_dir is not None:
        overrides.append(f"store_dir={args.store_dir}")
    if args.session_id is not None:
        overrides.append(f"session_id={args.session_id}")

    cmd = [sys.executable, "scripts/eval_model.py", *overrides, *rest]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
