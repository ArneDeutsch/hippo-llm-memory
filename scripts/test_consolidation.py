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
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional

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
    allow_tiny_test_model: bool
    uplift_mode: str
    min_uplift: float
    alpha: float


def _parse_args(argv: Optional[list[str]] = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pre", "post"], required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL"),
        help="Base model identifier or path",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--adapter", help="LoRA adapter path for post phase")
    parser.add_argument(
        "--pre_dir",
        help="Directory containing metrics from the pre phase (required for post phase)",
    )
    parser.add_argument(
        "--allow-tiny-test-model",
        action="store_true",
        help="Enable the tiny-gpt2 test model when MODEL is unset",
    )
    parser.add_argument(
        "--uplift-mode",
        choices=["fixed", "ci"],
        default="fixed",
        help="Gate mode: 'fixed' uses --min-uplift; 'ci' requires delta CI > 0",
    )
    parser.add_argument(
        "--min-uplift",
        type=float,
        default=0.05,
        help="Minimum EM uplift required in fixed mode",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CI mode (e.g., 0.05 for 95% CI)",
    )
    ns = parser.parse_args(argv)
    return ns  # type: ignore[return-value]


def _resolve_model(arg_model: str | None, allow_tiny: bool) -> str:
    m = (arg_model or os.environ.get("MODEL") or "").strip()
    if not m and allow_tiny:
        m = "models/tiny-gpt2"
    if not m:
        raise SystemExit(
            "Error: --model is empty and $MODEL is not set.\n",
            "Set --model (e.g., Qwen/Qwen2.5-1.5B-Instruct) or export MODEL.",
        )
    if m in {"tiny-gpt2", "models/tiny-gpt2"} and not allow_tiny:
        raise SystemExit(
            "Error: tiny-gpt2 is for tests only. Pass --allow-tiny-test-model to use it.",
        )
    return m


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
            "primary_em": "norm",
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
        if post_val is None:
            post_val = post_suite.get(f"pre_{base}")
        if isinstance(val, (int, float)) and isinstance(post_val, (int, float)):
            delta[base] = float(post_val) - float(val)
    return delta


def _collect_deltas(root: Path) -> List[float]:
    """Return all delta.em values under ``root`` for CI computation."""

    deltas: List[float] = []
    # metrics.json directly under root
    direct = root / "metrics.json"
    if direct.exists():
        d = io.read_json(direct).get("delta", {}).get("em")
        if isinstance(d, (int, float)):
            deltas.append(float(d))
    for path in root.glob("*/metrics.json"):
        d = io.read_json(path).get("delta", {}).get("em")
        if isinstance(d, (int, float)):
            deltas.append(float(d))
    return deltas


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = _parse_args(argv)
    args.model = _resolve_model(getattr(args, "model", None), args.allow_tiny_test_model)
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

        suite = data.get("suite")
        pre_em = pre_metrics.get("metrics", {}).get(suite, {}).get("pre_em", 0.0)
        post_em = data.get("metrics", {}).get(suite, {}).get("post_em", pre_em)
        uplift = delta.get("em", 0.0)
        print(f"pre_em={pre_em:.3f} post_em={post_em:.3f} delta={uplift:.3f}")

        if args.uplift_mode == "fixed":
            if uplift < args.min_uplift:
                raise RuntimeError(f"EM uplift < +{args.min_uplift:.2f} (got {uplift:.2f})")
        else:  # ci mode
            deltas = _collect_deltas(outdir.parent)
            if len(deltas) < 2:
                raise RuntimeError("--uplift-mode=ci requires at least two seeds")
            mean = statistics.mean(deltas)
            std = statistics.stdev(deltas)
            z = NormalDist().inv_cdf(1 - args.alpha / 2)
            se = std / math.sqrt(len(deltas))
            lower = mean - z * se
            upper = mean + z * se
            print(f"delta_em_mean={mean:.3f} ci=({lower:.3f}, {upper:.3f})")
            if lower <= 0.0:
                raise RuntimeError(
                    f"EM uplift CI includes 0 (mean={mean:.3f}, CI=({lower:.3f}, {upper:.3f}))"
                )
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return data


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
