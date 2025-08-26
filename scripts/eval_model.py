"""Thin wrapper around :mod:`hippo_mem.eval.harness` for CLI use.

The full evaluation harness lives in :mod:`hippo_mem.eval.harness`.  This
module re-exports the public API for backward compatibility and forwards
to :func:`hippo_mem.eval.harness.main` when executed as a script.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_mem.eval.harness import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalConfig,
    Task,
    _apply_model_defaults,
    _dataset_path,
    _evaluate,
    _load_preset,
    evaluate,
    evaluate_matrix,
    run_suite,
)
from hippo_mem.eval.harness import (
    main as harness_main,
)

__all__ = [
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "EvalConfig",
    "Task",
    "_apply_model_defaults",
    "_dataset_path",
    "_evaluate",
    "_load_preset",
    "evaluate",
    "evaluate_matrix",
    "run_suite",
    "main",
]


@hydra.main(version_base=None, config_path="../configs/eval", config_name="default")
def main(cfg: DictConfig) -> None:
    """Hydra entry point forwarding to :mod:`hippo_mem.eval.harness`."""

    harness_main(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
