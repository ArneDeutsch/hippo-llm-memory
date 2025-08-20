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

sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval_bench import _config_hash, _flatten_ablate, _git_sha, _init_modules


@dataclass
class Task:
    """Simple container for evaluation items."""

    prompt: str
    answer: str


def _dataset_path(suite: str, n: int, seed: int) -> Path:
    """Return path to a JSONL dataset for ``suite`` covering ``n`` items."""

    # Available canonical sizes.  Pick the smallest that covers ``n``.
    sizes = [50, 200, 1000]
    for size in sizes:
        if n <= size:
            return Path("data") / f"{suite}_{size}_{seed}.jsonl"
    return Path("data") / f"{suite}_{sizes[-1]}_{seed}.jsonl"


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
    tasks: Iterable[Task], modules: Dict[str, Dict[str, object]]
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Run the mock model over ``tasks`` and return rows and aggregate metrics."""

    rows: List[Dict[str, object]] = []
    correct = 0
    f1_total = 0.0
    for idx, item in enumerate(tasks):
        # Exercise memory modules for plumbing coverage only.
        if "episodic" in modules:
            store = modules["episodic"]["store"]
            store.recall(np.zeros(8, dtype="float32"), 1)
            hidden = torch.zeros(1, 1, 8)
            mem = torch.zeros(1, 1, 8)
            modules["episodic"]["adapter"](hidden, mem)
        # Mock model simply echoes the reference answer.
        pred = item.answer
        is_correct = int(pred == item.answer)
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

    # Metrics JSON
    metrics: Dict[str, object] = {
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "metrics": {"pre": pre_metrics},
    }
    if post_metrics is not None:
        metrics["metrics"]["post"] = post_metrics
        metrics["metrics"]["delta_em"] = post_metrics.get("em", 0.0) - pre_metrics.get("em", 0.0)
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
    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "prompt", "answer", "pred", "correct", "latency_ms", "flags"]
        )
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
    if cfg.get("dry_run"):
        cfg.n = min(cfg.n, 5)

    dataset = _dataset_path(cfg.suite, cfg.n, cfg.seed)
    tasks = _load_tasks(dataset, cfg.n)
    flat_ablate = _flatten_ablate(cfg.get("ablate"))
    modules = _init_modules(cfg.get("memory"), flat_ablate)

    pre_rows, pre_metrics = _evaluate(tasks, modules)
    post_rows = post_metrics = None

    for _ in range(int(cfg.get("replay", {}).get("cycles", 0))):
        _run_replay(modules, tasks)
        post_rows, post_metrics = _evaluate(tasks, modules)

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
