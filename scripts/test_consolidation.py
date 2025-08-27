"""Run pre/post consolidation tests with memory disabled.

This script evaluates a model on a suite with ``memory_off`` and optionally a
LoRA adapter attached. It supports two phases:

* ``pre``  – baseline evaluation with memory disabled.
* ``post`` – evaluation with a LoRA adapter; metrics are compared against a
  previous ``pre`` run and deltas are reported.

Example usage::

    # Pre-consolidation baseline
    python scripts/test_consolidation.py --phase pre --suite episodic --n 50 \
        --seed 1337 --model models/tiny-gpt2 --outdir runs/pre

    # Post-consolidation test
    python scripts/test_consolidation.py --phase post --suite episodic --n 50 \
        --seed 1337 --model models/tiny-gpt2 --adapter runs/adapter \
        --pre_dir runs/pre --outdir runs/post
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local modules are importable
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SCRIPT_DIR))
sys.path.append(str(_SCRIPT_DIR.parent))

import eval_model  # type: ignore  # noqa: E402

from hippo_mem.adapters.lora import load_adapter, merge_adapter  # noqa: E402
from hippo_mem.common import io  # noqa: E402


class Args(argparse.Namespace):
    phase: str
    suite: str
    n: int
    seed: int
    model: str
    outdir: str
    adapter: Optional[str]
    pre_dir: Optional[str]


def _parse_args(argv: Optional[list[str]] = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pre", "post"], required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("HF_MODEL_PATH", "models/tiny-gpt2"),
        help="Base model identifier or path",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--adapter", help="LoRA adapter path for post phase")
    parser.add_argument(
        "--pre_dir",
        help="Directory containing metrics from the pre phase (required for post phase)",
    )
    ns = parser.parse_args(argv)
    return ns  # type: ignore[return-value]


def _build_cfg(model_path: str, args: Args) -> Any:
    cfg = OmegaConf.create(
        {
            "suite": args.suite,
            "n": args.n,
            "seed": args.seed,
            "preset": "baselines/core",
            "model": model_path,
            "memory_off": True,
            "mode": "test",
        }
    )
    cfg = eval_model._load_preset(cfg)
    cfg = eval_model._apply_model_defaults(cfg)
    return cfg


def _prepare_model_with_adapter(
    base_model: str, adapter: str
) -> tuple[str, tempfile.TemporaryDirectory]:
    """Return path to a temp dir containing the base model with ``adapter`` merged."""

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = load_adapter(model, adapter)
    merged = merge_adapter(model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tmp = tempfile.TemporaryDirectory()
    merged.save_pretrained(tmp.name)
    tokenizer.save_pretrained(tmp.name)
    return tmp.name, tmp


def _compute_delta(pre_metrics: Dict[str, Any], post_metrics: Dict[str, Any]) -> Dict[str, float]:
    suite = post_metrics.get("suite")
    pre_suite = pre_metrics.get("metrics", {}).get(suite, {})
    post_suite = post_metrics.get("metrics", {}).get(suite, {})
    delta: Dict[str, float] = {}
    for key, val in pre_suite.items():
        if not key.startswith("pre_"):
            continue
        base = key[4:]
        post_val = post_suite.get(f"post_{base}")
        if isinstance(val, (int, float)) and isinstance(post_val, (int, float)):
            delta[base] = float(post_val) - float(val)
    return delta


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = _parse_args(argv)
    model_path = args.model
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if args.phase == "post":
        if not args.adapter or not args.pre_dir:
            raise ValueError("post phase requires --adapter and --pre_dir")
        model_path, tmp_dir = _prepare_model_with_adapter(args.model, args.adapter)

    cfg = _build_cfg(model_path, args)
    outdir = Path(args.outdir)
    eval_model.evaluate(cfg, outdir)

    result_path = outdir / "metrics.json"
    data = io.read_json(result_path)
    if args.phase == "post":
        pre_metrics = io.read_json(Path(args.pre_dir) / "metrics.json")
        delta = _compute_delta(pre_metrics, data)
        data["delta"] = delta
        io.atomic_write_json(result_path, data)
        if (
            args.suite == "episodic"
            and args.n == 50
            and args.seed == 1337
            and delta.get("em_raw", 0.0) < 0.20
        ):
            raise RuntimeError("EM uplift < +0.20 on episodic@50 seed=1337")
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return data


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
