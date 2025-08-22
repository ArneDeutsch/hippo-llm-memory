"""Evaluate a model on JSONL suites and write metrics and metadata.

This script implements the basic evaluation harness described in
``EVAL_PLAN.md``.  It loads a base model plus optional memory modules,
runs a suite of tasks loaded from JSONL files, computes simple metrics
and writes ``metrics.json``/``metrics.csv`` as well as ``meta.json`` to
an output directory.

Only a very small mock model is exercised in the unit tests.  The
implementation nevertheless mirrors the structure expected for the real
project and supports pre/post‑replay evaluation as well as ablation
flags.  Replay cycles simply feed the ground‑truth answers back into the
episodic store to mimic consolidation.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
from eval_bench import _config_hash, _flatten_ablate, _git_sha, _init_modules

from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.common.telemetry import registry
from hippo_mem.episodic.retrieval import episodic_retrieve_and_pack
from hippo_mem.relational.retrieval import relational_retrieve_and_pack
from hippo_mem.spatial.retrieval import spatial_retrieve_and_pack


@dataclass
class Task:
    """Simple container for evaluation items."""

    prompt: str
    answer: str


@dataclass
class EvalConfig:
    """Lightweight configuration for :func:`run_suite`."""

    suite: str
    n: int = 5
    seed: int = 0
    preset: str = "configs/eval/memory/hei_nw.yaml"
    model: str = "models/tiny-gpt2"
    max_new_tokens: int = 32
    replay_cycles: int = 0


def _dataset_path(suite: str, n: int, seed: int) -> Path:
    """Return path to a JSONL dataset for ``suite`` covering ``n`` items.

    The function falls back to the smallest canonical size that covers ``n``. If
    the exact file does not exist it searches for variants matching
    ``"{suite}_*_{size}_{seed}.jsonl"`` and returns the first match.
    """

    sizes = [50, 200, 1000]
    size = next((s for s in sizes if n <= s), sizes[-1])
    base = Path("data") / f"{suite}_{size}_{seed}.jsonl"
    if base.exists():
        return base
    candidates = sorted(Path("data").glob(f"{suite}_*_{size}_{seed}.jsonl"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Dataset not found; run scripts/build_datasets.py or check suite name")


def _load_tasks(path: Path, n: int) -> List[Task]:
    """Load the first ``n`` tasks from ``path``."""

    tasks: List[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(tasks) >= n:
                break
            obj = json.loads(line)
            tasks.append(Task(prompt=str(obj["prompt"]), answer=str(obj["answer"])))
    return tasks


def _f1(pred: str, truth: str) -> float:
    """Token-level F1 used for episodic and semantic suites."""

    pred_tokens = pred.split()
    truth_tokens = truth.split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _evaluate(
    tasks: Iterable[Task],
    modules: Dict[str, Dict[str, object]],
    tokenizer,
    model,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Generate predictions and compute metrics for ``tasks``."""

    rows: List[Dict[str, object]] = []
    correct = 0
    f1_total = 0.0
    for idx, item in enumerate(tasks):
        # Optional memory retrieval and adapter calls for plumbing coverage.
        if modules:
            hidden = torch.zeros(1, 1, 8)
            mems: List[MemoryTokens] = []
            if "episodic" in modules:
                store = modules["episodic"]["store"]
                adapter = modules["episodic"]["adapter"]
                proj = getattr(adapter, "proj", nn.Linear(store.dim, 8))
                adapter.proj = proj  # type: ignore[attr-defined]
                spec = TraceSpec(source="episodic", k=1)
                mems.append(episodic_retrieve_and_pack(hidden, spec, store, proj))
            if "relational" in modules:
                kg = modules["relational"]["kg"]
                adapter = modules["relational"]["adapter"]
                hidden_dim = hidden.size(-1)
                in_dim = getattr(kg, "dim", 0) or hidden_dim
                proj = getattr(adapter, "proj", nn.Linear(in_dim, hidden_dim))
                adapter.proj = proj  # type: ignore[attr-defined]
                spec = TraceSpec(source="relational", k=1)
                mems.append(relational_retrieve_and_pack(hidden, spec, kg, proj))
            if "spatial" in modules:
                graph = modules["spatial"]["map"]
                adapter = modules["spatial"]["adapter"]
                proj = getattr(adapter, "proj", nn.Linear(4, 8))
                adapter.proj = proj  # type: ignore[attr-defined]
                spec = TraceSpec(source="spatial")
                mems.append(spatial_retrieve_and_pack("origin", spec, graph, proj))
            if mems:
                tokens = torch.cat([m.tokens for m in mems], dim=1)
                mask = torch.cat([m.mask for m in mems], dim=1)
                mem = MemoryTokens(tokens=tokens, mask=mask)
                for mod in modules.values():
                    adapter = mod.get("adapter")
                    if adapter is not None:
                        adapter(hidden, memory=mem)

        enc = tokenizer(item.prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        is_correct = int(pred.strip() == item.answer)
        correct += is_correct
        f1_total += _f1(pred, item.answer)
        rows.append(
            {
                "idx": idx,
                "prompt": item.prompt,
                "answer": item.answer,
                "pred": pred,
                "correct": is_correct,
                "latency_ms": 0.0,
            }
        )
    n = len(rows)
    metrics = {"em": correct / n if n else 0.0, "f1": f1_total / n if n else 0.0}
    return rows, metrics


def _run_replay(modules: Dict[str, Dict[str, object]], tasks: Iterable[Task]) -> None:
    """Dummy replay loop that writes answers to the episodic store."""

    if "episodic" not in modules:
        return
    store = modules["episodic"]["store"]
    for task in tasks:
        key = np.ones(8, dtype="float32")
        store.write(key, task.answer)


def run_suite(
    cfg: EvalConfig,
) -> tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    """Execute a suite according to ``cfg`` and return rows and metrics."""

    registry.reset()

    base_cfg = OmegaConf.create(
        {
            "suite": cfg.suite,
            "n": cfg.n,
            "seed": cfg.seed,
            "model": cfg.model,
            "max_new_tokens": cfg.max_new_tokens,
        }
    )
    if cfg.preset:
        preset_cfg = OmegaConf.load(cfg.preset)
        base_cfg = OmegaConf.merge(base_cfg, preset_cfg)

    tasks = _load_tasks(_dataset_path(cfg.suite, cfg.n, cfg.seed), cfg.n)
    flat_ablate = _flatten_ablate(base_cfg.get("ablate"))
    modules = _init_modules(base_cfg.get("memory"), flat_ablate)

    model_path = to_absolute_path(str(base_cfg.model))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows, metrics = _evaluate(tasks, modules, tokenizer, model, int(base_cfg.max_new_tokens))

    for _ in range(int(cfg.replay_cycles)):
        _run_replay(modules, tasks)

    metrics_dict = {
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "metrics": {cfg.suite: metrics},
        "retrieval": registry.all_snapshots(),
    }
    return rows, metrics_dict, flat_ablate


def _write_outputs(
    outdir: Path,
    pre_rows: List[Dict[str, object]],
    pre_metrics: Dict[str, float],
    post_rows: Optional[List[Dict[str, object]]],
    post_metrics: Optional[Dict[str, float]],
    cfg: DictConfig,
    flat_ablate: Dict[str, object],
) -> None:
    """Persist metrics and metadata."""

    outdir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON - follow schema used by report.py
    suite_metrics: Dict[str, float] = {
        "pre_em": pre_metrics.get("em", 0.0),
        "pre_f1": pre_metrics.get("f1", 0.0),
    }
    if post_metrics is not None:
        suite_metrics.update(
            {
                "post_em": post_metrics.get("em", 0.0),
                "post_f1": post_metrics.get("f1", 0.0),
                "delta_em": post_metrics.get("em", 0.0) - pre_metrics.get("em", 0.0),
            }
        )
    metrics: Dict[str, object] = {
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "metrics": {cfg.suite: suite_metrics},
        "retrieval": registry.all_snapshots(),
    }
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f)

    # Per-task CSV
    csv_rows: List[Dict[str, object]] = []
    for row in pre_rows:
        row = dict(row)
        row["flags"] = "pre_replay"
        csv_rows.append(row)
    if post_rows is not None:
        for row in post_rows:
            row = dict(row)
            row["flags"] = "post_replay"
            csv_rows.append(row)
    retrieval_fields = {
        f"retrieval.{m}.{k}": v for m, snap in metrics["retrieval"].items() for k, v in snap.items()
    }
    for row in csv_rows:
        row.update(retrieval_fields)
    fieldnames = [
        "idx",
        "prompt",
        "answer",
        "pred",
        "correct",
        "latency_ms",
        "flags",
        *sorted(retrieval_fields),
    ]
    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # Metadata JSON
    meta = {
        "git_sha": _git_sha(),
        "model": cfg.get("model", "mock"),
        "config_hash": _config_hash(cfg),
        "ablate": flat_ablate,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "replay_cycles": cfg.get("replay", {}).get("cycles", 0),
    }
    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)


def _load_preset(cfg: DictConfig) -> DictConfig:
    """Merge the ``preset`` YAML into ``cfg`` if it exists."""

    preset = cfg.get("preset", "baselines/core")
    preset_path = Path(to_absolute_path("configs")) / "eval" / f"{preset}.yaml"
    if preset_path.exists():
        preset_cfg = OmegaConf.load(preset_path)
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, preset_cfg)
    return cfg


@hydra.main(version_base=None, config_path="../configs/eval", config_name="default")
def main(cfg: DictConfig) -> None:
    """CLI entry point for the evaluation harness."""

    cfg = _load_preset(cfg)
    cfg.n = cfg.get("n", 5)
    cfg.seed = cfg.get("seed", 0)
    cfg.model = cfg.get("model", "models/tiny-gpt2")
    cfg.max_new_tokens = cfg.get("max_new_tokens", 32)
    if cfg.get("dry_run"):
        cfg.n = min(cfg.n, 5)

    registry.reset()

    dataset = _dataset_path(cfg.suite, cfg.n, cfg.seed)
    tasks = _load_tasks(dataset, cfg.n)
    flat_ablate = _flatten_ablate(cfg.get("ablate"))
    modules = _init_modules(cfg.get("memory"), flat_ablate)

    model_path = to_absolute_path(str(cfg.model))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pre_rows, pre_metrics = _evaluate(tasks, modules, tokenizer, model, int(cfg.max_new_tokens))
    post_rows = post_metrics = None

    for _ in range(int(cfg.get("replay", {}).get("cycles", 0))):
        _run_replay(modules, tasks)
        post_rows, post_metrics = _evaluate(
            tasks, modules, tokenizer, model, int(cfg.max_new_tokens)
        )

    outdir_cfg = cfg.get("outdir")
    if outdir_cfg is not None:
        outdir = Path(to_absolute_path(str(outdir_cfg)))
    else:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        preset_path = Path(str(cfg.preset))
        if preset_path.parts and preset_path.parts[0] == "baselines":
            outdir = Path("runs") / date / preset_path.parts[0] / preset_path.name / cfg.suite
        else:
            outdir = Path("runs") / date / preset_path.name / cfg.suite

    _write_outputs(outdir, pre_rows, pre_metrics, post_rows, post_metrics, cfg, flat_ablate)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
