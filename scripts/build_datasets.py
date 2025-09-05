"""Build both legacy and memory-first dataset artifacts.

During the protocol migration (Milestone M6) we emit paired datasets so legacy
in‑prompt suites and new memory‑first suites can be compared side by side. For a
given ``--suite`` this script writes:

* ``datasets/<suite>/<suite>.jsonl`` – legacy in‑prompt data.
* ``datasets/<suite>_mem/`` or ``datasets/spatial_multi/`` – memory‑required
  teach/test splits.

Example:

```bash
python scripts/build_datasets.py --suite semantic --size 50 --seed 1337
```
"""

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
    """Build legacy and memory datasets for ``suite``."""

    legacy_dir = out_root / suite
    legacy_path = legacy_dir / f"{suite}.jsonl"
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
            str(legacy_path),
        ]
    )

    if suite == "spatial":
        mem_suite = "spatial_multi"
        mem_out = out_root / mem_suite
        _run_cli(
            [
                "--suite",
                mem_suite,
                "--size",
                str(size),
                "--seed",
                str(seed),
                "--profile",
                profile,
                "--out",
                str(mem_out),
            ]
        )
    else:
        mem_out = out_root / f"{suite}_mem"
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
                str(mem_out),
                "--require-memory",
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["semantic", "episodic_cross", "spatial"],
        nargs="+",
        default=["semantic", "episodic_cross", "spatial"],
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
