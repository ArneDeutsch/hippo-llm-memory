"""Thin wrapper around :mod:`hippo_mem.eval.harness` for CLI use.

The full evaluation harness lives in :mod:`hippo_mem.eval.harness`.  This
module re-exports the public API for backward compatibility and forwards
to :func:`hippo_mem.eval.harness.main` when executed as a script.

It exposes the ``teach``, ``replay`` and ``test`` modes as well as
``store_dir``/``session_id`` persistence flags used by the evaluation
protocol.
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
    if cfg.mode in ("replay", "test"):
        if not getattr(cfg, "store_dir", None) or not getattr(cfg, "session_id", None):
            raise SystemExit("Error: --store_dir and --session_id are required for this mode.")
        from hippo_mem.utils.stores import assert_store_exists

        root = Path(str(cfg.store_dir))
        base = root.parent if root.name == "hei_nw" else root
        assert_store_exists(str(base), str(cfg.session_id), kind="episodic")
        cfg.store_dir = str(base / "hei_nw")
    elif getattr(cfg, "store_dir", None):
        root = Path(str(cfg.store_dir))
        cfg.store_dir = str(root if root.name == "hei_nw" else root / "hei_nw")

    harness_main(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
