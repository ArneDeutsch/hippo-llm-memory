import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_eval.reporting.report import sanity_sweep
from hippo_eval.reporting.summarize import summarize_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, help="root directory for legacy mode")
    parser.add_argument("--out", type=Path, help="output directory for legacy mode")
    parser.add_argument("--run-id", help="run identifier to sweep", default=None)
    parser.add_argument("--runs-dir", default="runs", help="base directory containing runs")
    args = parser.parse_args(argv)
    if args.run_id:
        sanity_sweep(Path(args.runs_dir), args.run_id)
        return 0
    if args.root is None or args.out is None:
        parser.error("root and --out are required when --run-id is not set")
    summarize_runs(args.root, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
