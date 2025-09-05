"""Build memory-first datasets for evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run_cli(args: list[str]) -> None:
    """Invoke ``datasets_cli.py`` with ``args``."""
    cmd = [sys.executable, str(ROOT / "datasets_cli.py"), *args]
    subprocess.run(cmd, check=True)


def _build_suite(suite: str, size: int, seed: int, profile: str, out_root: Path) -> None:
    """Build memory-first dataset for ``suite``."""
    out_dir = out_root / suite
    _run_cli(
        [
            "--suite",
            suite,
            "--size",
            str(size),
            "--seed",
            str(seed),
            "--profile",
            profile,
            "--out",
            str(out_dir),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["semantic_mem", "episodic_cross_mem", "spatial_multi"],
        nargs="+",
        default=["semantic_mem", "episodic_cross_mem", "spatial_multi"],
    )
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=["easy", "default", "hard"], default="default")
    parser.add_argument("--out-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    for suite in args.suite:
        _build_suite(suite, args.size, args.seed, args.profile, args.out_root)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
