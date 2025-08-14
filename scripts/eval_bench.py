"""Light‑weight evaluation harness used in tests and CI.

This module provides a tiny yet functional implementation of the evaluation
workflow described in :mod:`EVAL_PLAN.md`.  It does **not** attempt to run any
real language model – instead predictions are taken to be equal to the ground
truth answers.  The goal is to exercise the metric plumbing and file layout so
that higher level tooling can be validated without expensive model inference.

Example usage from the command line::

    python scripts/eval_bench.py suite=episodic preset=baselines/core n=5 seed=0

The resulting ``metrics.json``/``metrics.csv``/``meta.json`` files are written
to ``runs/<date>/<preset>/<suite>/`` by default or to the directory supplied via
``outdir=...``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import yaml
from build_datasets import SUITE_TO_GENERATOR
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.map import PlaceGraph


def _git_sha() -> str:
    """Return the current git commit SHA, or ``unknown`` if unavailable."""

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:  # pragma: no cover - extremely unlikely in tests
        return "unknown"


def _init_modules(memory: Optional[List[str]]) -> Dict[str, object]:
    """Initialise memory modules based on ``memory`` names."""

    modules: Dict[str, object] = {}
    if not memory:
        return modules
    for name in memory:
        if name == "hei_nw":
            store = EpisodicStore(dim=8)
            # seed with a dummy entry
            store.write(np.ones(8, dtype="float32"), "dummy")
            modules["hei_nw"] = store
        elif name == "sgc_rss":
            kg = KnowledgeGraph()
            kg.upsert("a", "rel", "b", "a rel b")
            modules["sgc_rss"] = kg
        elif name == "smpd":
            g = PlaceGraph()
            g.observe("a")
            g.observe("b")
            g.connect("a", "b")
            modules["smpd"] = g
    return modules


def _flatten_ablate(ablate: Optional[DictConfig | str]) -> Dict[str, object]:
    """Flatten ablation flags into ``{"a.b": value}`` form."""

    if ablate in (None, {}, ""):
        return {}
    flat: Dict[str, object] = {}
    if isinstance(ablate, str):
        parts = [p for p in ablate.split(",") if p]
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                flat[key.strip()] = yaml.safe_load(val.strip())
        return flat

    data = OmegaConf.to_container(ablate, resolve=True)

    def _recurse(d: Dict[str, object], prefix: str = "") -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _recurse(v, f"{prefix}{k}.")
            else:
                flat[f"{prefix}{k}"] = v

    _recurse(data)
    return flat


def evaluate(cfg: DictConfig, outdir: Path) -> None:
    """Run evaluation for ``cfg.suite`` and write metrics to ``outdir``."""

    suite = cfg.suite
    n = cfg.n
    seed = cfg.seed
    preset = cfg.preset

    generator = SUITE_TO_GENERATOR[suite]
    tasks = generator(n, seed)

    modules = _init_modules(cfg.get("memory"))

    adapter: EpisodicAdapter | None = None
    if "hei_nw" in modules:
        adapter_cfg = AdapterConfig(hidden_size=8, num_heads=1, enabled=True)
        adapter = EpisodicAdapter(adapter_cfg)

    rows: List[Dict[str, object]] = []
    correct = 0
    total_tokens = 0
    for idx, item in enumerate(tasks):
        # Exercise retrieval APIs for smoke coverage only.
        if "hei_nw" in modules:
            store = modules["hei_nw"]
            traces = store.recall(np.zeros(8, dtype="float32"), 1)
            if adapter is not None:
                mem = torch.tensor([t.key for t in traces], dtype=torch.float32).unsqueeze(0)
                hidden = torch.zeros(1, 1, 8)
                adapter(hidden, mem)
        if "sgc_rss" in modules:
            modules["sgc_rss"].retrieve(np.zeros(8, dtype="float32"))
        if "smpd" in modules:
            modules["smpd"].plan("a", "b")

        prompt = str(item["prompt"])
        answer = str(item["answer"])
        pred = str(item.get("pred", answer))  # echo model

        is_correct = int(pred.strip().lower() == answer.strip().lower())
        correct += is_correct
        total_tokens += len(prompt.split()) + len(answer.split())
        rows.append(
            {
                "idx": idx,
                "prompt": prompt,
                "answer": answer,
                "pred": pred,
                "correct": is_correct,
                "latency_ms": 0.0,
                "flags": "pre_replay",
            }
        )

    em = correct / n if n else 0.0
    metrics = {
        "suite": suite,
        "n": n,
        "seed": seed,
        "preset": preset,
        "metrics": {suite: {"em": em}, "compute": {"tokens": total_tokens}},
    }

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f)

    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "prompt",
                "answer",
                "pred",
                "correct",
                "latency_ms",
                "flags",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    cfg_hash = hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")).hexdigest()
    meta = {
        "git_sha": _git_sha(),
        "model": cfg.get("model", "mock"),
        "config_hash": cfg_hash,
        "ablate": _flatten_ablate(cfg.get("ablate")),
    }
    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)


@hydra.main(version_base=None, config_path="../configs/eval", config_name="default")
def main(cfg: DictConfig) -> None:
    """CLI entry point for the evaluation harness."""

    cfg.preset = cfg.get("preset", "baselines/core")
    cfg.seed = cfg.get("seed", 0)
    cfg.n = cfg.get("n", 5)
    if cfg.get("dry_run"):
        cfg.n = min(cfg.n, 5)
    outdir: Optional[str] = cfg.get("outdir")
    if outdir is not None:
        outdir_path = Path(to_absolute_path(outdir))
    else:
        outdir_path = (
            Path("runs")
            / datetime.utcnow().strftime("%Y%m%d")
            / str(cfg.preset).replace("/", "_")
            / cfg.suite
        )
    evaluate(cfg, outdir_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
