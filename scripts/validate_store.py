"""Validate expected persisted store layout before replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

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
    args = parser.parse_args()

    try:
        path = validate_store(
            run_id=args.run_id, algo=args.algo, kind=args.kind, preset=args.preset
        )
    except (FileExistsError, FileNotFoundError) as err:  # pragma: no cover - tested via CLI
        print(err, file=sys.stderr)
        raise SystemExit(1) from err
    if path is None:
        print(f"OK: no store for baseline {args.preset}")
    else:
        print(f"OK: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
