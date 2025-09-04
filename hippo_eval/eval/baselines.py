"""Run baseline evaluation matrix over presets, suites, sizes and seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from hippo_mem.utils import validate_run_id

PRESETS = ["baselines/core", "baselines/span_short", "baselines/rag", "baselines/longctx"]
SUITES = [
    "episodic",
    "semantic",
    "spatial",
    "episodic_multi",
    "episodic_cross",
    "episodic_capacity",
]
SIZES = [50, 200, 1000]
SEEDS = [1337, 2025, 4242]


def _sha256_file(path: Path) -> str:
    """Return SHA256 digest of ``path``."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_dataset(suite: str, size: int, seed: int) -> None:
    """Ensure dataset exists and matches recorded checksum."""

    data_path = Path("data") / suite / f"{size}_{seed}.jsonl"
    checksum_path = data_path.parent / "checksums.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset {data_path}")
    if not checksum_path.exists():
        raise FileNotFoundError(f"Missing checksum file {checksum_path}")
    with checksum_path.open("r", encoding="utf-8") as f:
        checksums = json.load(f)
    digest = _sha256_file(data_path)
    if checksums.get(data_path.name) != digest:
        raise RuntimeError(f"Checksum mismatch for {data_path}")


def _run(cmd: Iterable[str]) -> None:
    """Execute subprocess ``cmd`` and fail on non-zero return."""

    subprocess.run(list(cmd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--presets", nargs="*", default=PRESETS)
    parser.add_argument("--suites", nargs="*", default=SUITES)
    parser.add_argument("--sizes", nargs="*", type=int, default=SIZES)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = parser.parse_args()

    run_id = validate_run_id(args.run_id)

    for preset in args.presets:
        for suite in args.suites:
            for size in args.sizes:
                for seed in args.seeds:
                    _validate_dataset(suite, size, seed)
                    outdir = Path("runs") / run_id / preset / suite / f"{size}_{seed}"
                    cmd = [
                        sys.executable,
                        "-m",
                        "hippo_eval.bench",
                        f"suite={suite}",
                        f"preset={preset}",
                        f"n={size}",
                        f"seed={seed}",
                        f"run_id={run_id}",
                        f"outdir={outdir}",
                    ]
                    _run(cmd)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
