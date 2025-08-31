"""Validate expected persisted store layout before replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_mem.utils.stores import assert_store_exists
from scripts.store_paths import derive


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", help="Run identifier; defaults to $RUN_ID or $DATE env vars")
    parser.add_argument("--algo", default="hei_nw", help="Memory algorithm identifier")
    parser.add_argument("--kind", default="episodic", help="Store kind to validate")
    args = parser.parse_args()

    layout = derive(run_id=args.run_id, algo=args.algo)
    try:
        path = assert_store_exists(
            str(layout.base_dir), layout.session_id, args.algo, kind=args.kind
        )
    except FileNotFoundError as err:  # pragma: no cover - exercised in tests
        print(err, file=sys.stderr)
        raise SystemExit(1) from err
    print(f"OK: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
