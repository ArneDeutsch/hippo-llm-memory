# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Run pre/post consolidation tests with memory disabled.

This script evaluates a model on a suite with ``memory_off`` and optionally a
LoRA adapter attached. It supports two phases:

* ``pre``  – baseline evaluation with memory disabled.
* ``post`` – evaluation with a LoRA adapter; metrics are compared against a
  previous ``pre`` run and deltas are reported.

Example usage::

    # Pre-consolidation baseline
    python scripts/test_consolidation.py --phase pre --suite episodic --n 50 \
        --seed 1337 --model hippo_mem.testing.fake_hf.FAKE_MODEL_ID --outdir runs/pre

    # Post-consolidation test
    python scripts/test_consolidation.py --phase post --suite episodic --n 50 \
        --seed 1337 --model hippo_mem.testing.fake_hf.FAKE_MODEL_ID --adapter runs/adapter \
        --pre_dir runs/pre --outdir runs/post
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import hippo_eval.eval.harness as eval_model
from hippo_mem.adapters.lora import load_adapter, merge_adapter
from hippo_mem.common import io
from hippo_mem.testing.fake_hf import FAKE_MODEL_ID, is_fake_model_id, resolve_fake_model_id

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


DEFAULT_MIN_EM_UPLIFT = {"episodic": 0.05, "semantic": 0.02}


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
    min_em_uplift: Optional[float]
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
        help="Enable the fake test model when MODEL is unset",
    )
    parser.add_argument(
        "--uplift-mode",
        choices=["fixed", "ci"],
        default="fixed",
        help="Gate mode: 'fixed' uses --min-em-uplift; 'ci' requires delta CI > 0",
    )
    parser.add_argument(
        "--min-em-uplift",
        type=float,
        help="Minimum EM uplift required in fixed mode; defaults depend on suite",
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
    raw = (arg_model or os.environ.get("MODEL") or "").strip()
    if not raw and allow_tiny:
        raw = FAKE_MODEL_ID
    if not raw:
        raise SystemExit(
            "Error: --model is empty and $MODEL is not set.\n",
            "Set --model (e.g., Qwen/Qwen2.5-1.5B-Instruct) or export MODEL.",
        )
    if is_fake_model_id(raw):
        if not allow_tiny:
            raise SystemExit(
                f"Error: {FAKE_MODEL_ID} is for tests only. Pass --allow-tiny-test-model to use it.",
            )
        return FAKE_MODEL_ID
    return raw


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
    try:
        eval_model._dataset_path(cfg.suite, cfg.n, cfg.seed, cfg.get("dataset_profile"), cfg.mode)
    except FileNotFoundError:
        size = 50 if cfg.n <= 50 else cfg.n
        outdir = Path("data") / cfg.suite
        outdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hippo_eval.datasets.cli",
                "--suite",
                cfg.suite,
                "--size",
                str(size),
                "--seed",
                str(cfg.seed),
                "--out",
                str(outdir),
            ],
            check=True,
        )
    return cfg


def _prepare_model_with_adapter(
    base_model: str, adapter: str
) -> tuple[str, tempfile.TemporaryDirectory]:
    """Return path to a temp dir containing the base model with ``adapter`` merged."""

    resolved = resolve_fake_model_id(base_model) or base_model
    model = AutoModelForCausalLM.from_pretrained(resolved)
    model = load_adapter(model, adapter)
    merged = merge_adapter(model)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
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


def _assert_lineage(pre_metrics: Dict[str, Any], adapter: str) -> None:
    """Raise ``RuntimeError`` if consolidation lineage is invalid."""

    suite = pre_metrics.get("suite")
    pre_suite = pre_metrics.get("metrics", {}).get(suite, {})
    pre_em = pre_suite.get("pre_em")
    if not isinstance(pre_em, (int, float)) or math.isnan(float(pre_em)):
        raise RuntimeError("missing pre_em in pre metrics; run with pre-metrics enabled")

    run_path = Path(adapter).resolve()
    store_dir = None
    for parent in [run_path] + list(run_path.parents):
        candidate = parent / "stores"
        if candidate.exists():
            store_dir = candidate
            break
    if store_dir is None:
        store_dir = run_path.parent / "stores"
    metas = list(store_dir.rglob("store_meta.json"))
    if not metas:
        raise RuntimeError(f"missing store_meta.json under {store_dir}")
    for meta_path in metas:
        meta = io.read_json(meta_path)
        if meta.get("source") == "stub":
            raise RuntimeError(f"store {meta_path} marked as stub")
        if int(meta.get("replay_samples", 0)) < 1:
            raise RuntimeError(f"store {meta_path} has replay_samples < 1")


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = _parse_args(argv)
    if args.min_em_uplift is None:
        args.min_em_uplift = DEFAULT_MIN_EM_UPLIFT.get(args.suite, 0.05)
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
        _assert_lineage(pre_metrics, args.adapter)
        delta = _compute_delta(pre_metrics, data)
        data["delta"] = delta
        io.atomic_write_json(result_path, data)

        suite = data.get("suite")
        pre_em = pre_metrics.get("metrics", {}).get(suite, {}).get("pre_em", 0.0)
        post_em = data.get("metrics", {}).get(suite, {}).get("post_em", pre_em)
        uplift = delta.get("em", 0.0)

        print(f"pre_em={pre_em:.3f} post_em={post_em:.3f} delta={uplift:.3f}")

        deltas = _collect_deltas(outdir.parent)
        if len(deltas) >= 2:
            if args.uplift_mode == "fixed":
                print(f"detected {len(deltas)} seeds; switching to CI mode")
            mean = statistics.mean(deltas)
            std = statistics.stdev(deltas)
            z = NormalDist().inv_cdf(1 - args.alpha / 2)
            se = std / math.sqrt(len(deltas))
            lower = mean - z * se
            upper = mean + z * se
            print(f"delta_em_mean={mean:.3f} ci=({lower:.3f}, {upper:.3f})")
            if lower <= 0.0:
                raise RuntimeError(
                    f"EM uplift CI includes 0 (mean={mean:.3f}, CI=({lower:.3f}, {upper:.3f}))",
                )
        else:
            if args.uplift_mode == "ci":
                raise RuntimeError("--uplift-mode=ci requires at least two seeds")
            if uplift < args.min_em_uplift:
                raise RuntimeError(
                    f"EM uplift < +{args.min_em_uplift:.2f} (got {uplift:.2f})",
                )
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return data


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
