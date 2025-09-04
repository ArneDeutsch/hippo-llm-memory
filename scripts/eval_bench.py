"""Thin wrapper exposing the lightweight evaluation bench CLI.

The implementation resides in :mod:`hippo_eval.eval.bench`.  This script
forwards to :func:`hippo_eval.eval.bench.main` for compatibility with
existing tests and command line invocations.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_eval.eval.bench import (
    _config_hash,
    _git_sha,
    _init_modules,
    write_outputs,
)
from hippo_eval.eval.bench import (
    main as bench_main,
)

__all__ = [
    "_config_hash",
    "_git_sha",
    "_init_modules",
    "write_outputs",
    "main",
]


@hydra.main(version_base=None, config_path="../configs/eval", config_name="default")
def main(cfg: DictConfig) -> None:
    """Hydra entry point that forwards to :mod:`hippo_eval.eval.bench`."""

    bench_main(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
