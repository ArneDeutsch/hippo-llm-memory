"""Thin wrapper around :mod:`hippo_eval.eval.harness` for CLI use.

The full evaluation harness lives in :mod:`hippo_eval.eval.harness`.  This
module re-exports the public API for backward compatibility and forwards
to :func:`hippo_eval.eval.harness.main` when executed as a script.

It exposes the ``teach``, ``replay`` and ``test`` modes as well as
``store_dir``/``session_id`` persistence flags used by the evaluation
protocol.  In ``replay`` mode the harness now evaluates the suite after
consolidation and updates ``metrics.json`` with ``post_*`` and
``delta_*`` fields.
"""

import os
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

# expose oracle metrics via a boolean flag
if "--oracle" in sys.argv:
    idx = sys.argv.index("--oracle")
    sys.argv[idx] = "compute.oracle=true"

from hippo_eval.eval.harness import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Task,
    _apply_model_defaults,
    _dataset_path,
    _evaluate,
    _load_preset,
    evaluate,
    evaluate_matrix,
)
from hippo_eval.eval.harness import main as harness_main
from hippo_mem.utils.stores import StoreLayout, assert_store_exists
from hippo_mem.utils.stores import derive as derive_store_layout

__all__ = [
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "Task",
    "_apply_model_defaults",
    "_dataset_path",
    "_evaluate",
    "_load_preset",
    "evaluate",
    "evaluate_matrix",
    "main",
]


def _infer_algo(preset: str | None) -> str:
    """Derive algorithm key from memory preset."""

    p = Path(str(preset)) if preset else None
    if p and len(p.parts) >= 2 and p.parts[0] == "memory":
        return p.parts[1]
    return "hei_nw"


def _store_kind(algo: str) -> str:
    """Return store kind for a given algorithm."""

    return {"sgc_rss": "kg", "smpd": "spatial"}.get(algo, "episodic")


def _normalize_store_dir(store_dir: str, algo: str) -> str:
    """Return base ``store_dir`` without trailing algorithm suffix."""

    root = Path(store_dir)
    return str(root.parent if root.name == algo else root)


def _resolve_layout(cfg: DictConfig, algo: str) -> StoreLayout:
    """Resolve persisted store layout and update ``cfg`` in-place."""

    if cfg.get("store_dir") and cfg.get("session_id"):
        base = _normalize_store_dir(str(cfg.store_dir), algo)
        base_dir = Path(base)
        algo_dir = base_dir / algo
        with open_dict(cfg):
            cfg.store_dir = str(algo_dir)
        return StoreLayout(base_dir, algo_dir, str(cfg.session_id))
    layout = derive_store_layout(algo=algo)
    with open_dict(cfg):
        cfg.store_dir = cfg.get("store_dir") or str(layout.base_dir)
        cfg.session_id = cfg.get("session_id") or layout.session_id
    return layout


def _ensure_populated(store_path: Path | None, cfg: DictConfig) -> None:
    """Raise ``FileNotFoundError`` if ``store_path`` has no data."""

    if store_path is None or bool(cfg.get("dry_run")):
        return
    if store_path.exists():
        with store_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    return
    raise FileNotFoundError(
        "empty store: "
        f"{store_path} — populate via:\n  "
        f"python scripts/eval_model.py --mode teach +suite={cfg.suite} --run-id {cfg.run_id}"
    )


@hydra.main(version_base=None, config_path="../configs/eval", config_name="default")
def main(cfg: DictConfig) -> None:
    """Hydra entry point forwarding to :mod:`hippo_eval.eval.harness`."""

    algo = _infer_algo(cfg.get("preset"))
    store_kind = _store_kind(algo)

    # propagate mode to memory config for downstream components
    with open_dict(cfg):
        if cfg.get("memory") is None:
            cfg.memory = {}
        cfg.memory["mode"] = cfg.mode

    if cfg.mode not in {"teach", "replay", "test"}:
        harness_main(cfg)
        return

    layout: StoreLayout | None = None
    if cfg.get("store_dir") and cfg.get("session_id"):
        layout = _resolve_layout(cfg, algo)

    if cfg.mode == "replay" and layout is not None:
        store_path = assert_store_exists(
            str(layout.base_dir), str(cfg.session_id), algo, kind=store_kind
        )
        _ensure_populated(store_path, cfg)
    elif cfg.mode == "test":
        needs_store = (
            str(cfg.get("preset", "")).startswith("memory/")
            and str(cfg.get("mode", "test")) in {"test", "replay"}
            and layout is not None
        )
        if needs_store:
            from hippo_mem.utils.stores import validate_store

            maybe_store = validate_store(
                run_id=str(cfg.get("run_id") or os.getenv("RUN_ID") or ""),
                preset=str(cfg.preset),
                algo=algo,
                kind=store_kind,
                store_dir=str(layout.base_dir),
                session_id=str(cfg.get("session_id")),
            )
            if maybe_store is not None:
                _ensure_populated(maybe_store, cfg)
    elif cfg.get("store_dir"):
        cfg.store_dir = _normalize_store_dir(str(cfg.store_dir), algo)

    harness_main(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
