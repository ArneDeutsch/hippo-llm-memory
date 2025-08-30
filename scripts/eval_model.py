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
from hippo_mem.eval.harness import main as harness_main

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

    def _infer_algo(preset: str | None) -> str:
        p = Path(str(preset)) if preset else None
        if p and len(p.parts) >= 2 and p.parts[0] == "memory":
            return p.parts[1]
        return "hei_nw"

    algo = _infer_algo(cfg.get("preset"))

    if cfg.mode in ("replay", "test"):
        if not getattr(cfg, "store_dir", None) or not getattr(cfg, "session_id", None):
            raise SystemExit("Error: --store_dir and --session_id are required for this mode.")
        from hippo_mem.utils.stores import assert_store_exists

        root = Path(str(cfg.store_dir))
        if root.name == algo:
            print(
                f"Warning: store_dir already ends with '{algo}'; not appending.",
                file=sys.stderr,
            )
            base_dir = root.parent
            cfg.store_dir = str(root)
        else:
            base_dir = root
            cfg.store_dir = str(root / algo)
        assert_store_exists(str(base_dir), str(cfg.session_id), algo, kind="episodic")
    elif getattr(cfg, "store_dir", None):
        root = Path(str(cfg.store_dir))
        if root.name == algo:
            print(
                f"Warning: store_dir already ends with '{algo}'; not appending.",
                file=sys.stderr,
            )
            cfg.store_dir = str(root)
        else:
            cfg.store_dir = str(root / algo)

    harness_main(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
