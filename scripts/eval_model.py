"""Thin wrapper around :mod:`hippo_mem.eval.harness` for CLI use.

The full evaluation harness lives in :mod:`hippo_mem.eval.harness`.  This
module re-exports the public API for backward compatibility and forwards
to :func:`hippo_mem.eval.harness.main` when executed as a script.

It exposes the ``teach``, ``replay`` and ``test`` modes as well as
``store_dir``/``session_id`` persistence flags used by the evaluation
protocol.  In ``replay`` mode the harness now evaluates the suite after
consolidation and updates ``metrics.json`` with ``post_*`` and
``delta_*`` fields.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

sys.path.append(str(Path(__file__).resolve().parent.parent))

# allow `--mode test` style flags by translating to Hydra syntax
if "--mode" in sys.argv:
    idx = sys.argv.index("--mode")
    if idx + 1 < len(sys.argv):
        val = sys.argv[idx + 1]
        sys.argv[idx] = f"mode={val}"
        del sys.argv[idx + 1]

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
from hippo_mem.utils.stores import StoreLayout, assert_store_exists
from hippo_mem.utils.stores import derive as derive_store_layout

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

    def _store_kind(a: str) -> str:
        return {"sgc_rss": "kg", "smpd": "spatial"}.get(a, "episodic")

    store_kind = _store_kind(algo)

    # propagate mode to memory config for downstream components
    with open_dict(cfg):
        if cfg.get("memory") is None:
            cfg.memory = {}
        cfg.memory["mode"] = cfg.mode

    layout: StoreLayout | None = None
    if cfg.mode == "replay":
        if not getattr(cfg, "store_dir", None) or not getattr(cfg, "session_id", None):
            layout = derive_store_layout(algo=algo)
            cfg.store_dir = cfg.store_dir or str(layout.algo_dir)
            cfg.session_id = cfg.session_id or layout.session_id
        else:
            root = Path(str(cfg.store_dir))
            base_dir = root.parent if root.name == algo else root
            cfg.store_dir = str(root if root.name == algo else root / algo)
            layout = StoreLayout(
                base_dir=base_dir, algo_dir=Path(cfg.store_dir), session_id=str(cfg.session_id)
            )
        store_path = assert_store_exists(
            str(layout.base_dir), str(cfg.session_id), algo, kind=store_kind
        )
        if store_path is not None:
            has_data = False
            if store_path.exists():
                with store_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            has_data = True
                            break
            if not has_data:
                raise FileNotFoundError(
                    "empty store: "
                    f"{store_path} — populate via:\n  "
                    f"python scripts/eval_model.py --mode teach +suite={cfg.suite} --run-id {cfg.run_id}"
                )
    elif cfg.mode == "test" and (
        getattr(cfg, "store_dir", None) or getattr(cfg, "session_id", None)
    ):
        if getattr(cfg, "store_dir", None) and getattr(cfg, "session_id", None):
            root = Path(str(cfg.store_dir))
            base_dir = root.parent if root.name == algo else root
            cfg.store_dir = str(root if root.name == algo else root / algo)
            layout = StoreLayout(
                base_dir=base_dir, algo_dir=Path(cfg.store_dir), session_id=str(cfg.session_id)
            )
        else:
            layout = derive_store_layout(algo=algo)
            if getattr(cfg, "store_dir", None) is None:
                cfg.store_dir = str(layout.algo_dir)
            if getattr(cfg, "session_id", None) is None:
                cfg.session_id = layout.session_id
        store_path = assert_store_exists(
            str(layout.base_dir), str(cfg.session_id), algo, kind=store_kind
        )
        if store_path is not None:
            has_data = False
            if store_path.exists():
                with store_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            has_data = True
                            break
            if not has_data:
                raise FileNotFoundError(
                    "empty store: "
                    f"{store_path} — populate via:\n  "
                    f"python scripts/eval_model.py --mode teach +suite={cfg.suite} --run-id {cfg.run_id}"
                )
    elif getattr(cfg, "store_dir", None):
        root = Path(str(cfg.store_dir))
        cfg.store_dir = str(root if root.name == algo else root / algo)

    harness_main(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
