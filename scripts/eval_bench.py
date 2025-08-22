"""Light‑weight evaluation harness used in tests and CI.

This module provides a tiny yet functional implementation of the evaluation
workflow described in :mod:`EVAL_PLAN.md`.  It does **not** attempt to run any
real language model – instead predictions are taken to be equal to the ground
truth answers.  The goal is to exercise the metric plumbing and file layout so
that higher level tooling can be validated without expensive model inference.

Example usage from the command line::

    python scripts/eval_bench.py suite=episodic preset=baselines/core n=5 seed=0

For a complete sweep across suites, dataset sizes and seeds use::

    python scripts/eval_bench.py +run_matrix=true preset=memory/hei_nw

The resulting ``metrics.json``/``metrics.csv``/``meta.json`` files are written
to ``runs/<date>/<preset>/<suite>/`` for baseline presets such as
``baselines/core``.  For memory presets (``memory/hei_nw`` etc.) the files are
stored under ``runs/<date>/<preset_name>/<suite>/`` where ``preset_name`` is the
final path component.  A custom root directory can be supplied via
``outdir=...``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import yaml
from build_datasets import SUITE_TO_GENERATOR
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
from types import SimpleNamespace

import torch

from hippo_mem.adapters import (
    EpisodicMemoryAdapter,
    RelationalMemoryAdapter,
    SpatialMemoryAdapter,
)
from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic.adapter import AdapterConfig
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.adapter import AdapterConfig as SpatialAdapterConfig
from hippo_mem.spatial.place_graph import PlaceGraph


def _git_sha() -> str:
    """Return the current git commit SHA, or ``unknown`` if unavailable."""

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:  # pragma: no cover - extremely unlikely in tests
        return "unknown"


def _config_hash(cfg: DictConfig) -> str:
    """Return a SHA256 hash of the resolved Hydra config."""

    yaml_dump = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(yaml_dump.encode("utf-8")).hexdigest()


def _init_modules(
    memory: Optional[object], ablate: Dict[str, object]
) -> Dict[str, Dict[str, object]]:
    """Initialise memory modules based on ``memory`` config.

    ``memory`` may be a single name, list of names or a dict with keys
    ``episodic``, ``relational`` and ``spatial``.  ``ablate`` contains flattened
    ablation flags which can disable optional components such as Hopfield or
    product quantisation.
    """

    modules: Dict[str, Dict[str, object]] = {}
    if not memory:
        return modules

    def _add_episodic() -> None:
        hopfield = bool(ablate.get("memory.episodic.hopfield", True))
        pq = bool(ablate.get("memory.episodic.pq", True))
        store = EpisodicStore(dim=8, config={"hopfield": hopfield, "pq": pq})
        store.write(np.ones(8, dtype="float32"), "dummy")
        adapter_cfg = AdapterConfig(hidden_size=8, num_heads=1, enabled=True)
        modules["episodic"] = {"store": store, "adapter": EpisodicMemoryAdapter(adapter_cfg)}

    def _add_relational() -> None:
        kg = KnowledgeGraph()
        kg.upsert("a", "rel", "b", "a rel b")
        modules["relational"] = {"kg": kg, "adapter": RelationalMemoryAdapter()}

    def _add_spatial() -> None:
        g = PlaceGraph()
        g.observe("a")
        g.observe("b")
        g.connect("a", "b")
        spat_cfg = SpatialAdapterConfig(hidden_size=8, num_heads=1, enabled=True)
        modules["spatial"] = {"map": g, "adapter": SpatialMemoryAdapter(spat_cfg)}

    if isinstance(memory, str):
        memory = [memory]

    if isinstance(memory, list):
        if "hei_nw" in memory:
            _add_episodic()
        if "sgc_rss" in memory:
            _add_relational()
        if "smpd" in memory:
            _add_spatial()
        return modules

    mem_dict = OmegaConf.to_container(memory, resolve=True)
    if mem_dict.get("episodic") is not None:
        _add_episodic()
    if mem_dict.get("relational") is not None:
        _add_relational()
    if mem_dict.get("spatial") is not None:
        _add_spatial()
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


def generate_tasks(suite: str, n: int, seed: int) -> List[Dict[str, object]]:
    """Return ``n`` tasks for ``suite`` using ``seed``."""

    generator = SUITE_TO_GENERATOR[suite]
    return generator(n, seed)


def _eval_tasks(
    tasks: List[Dict[str, object]],
    modules: Dict[str, Dict[str, object]],
    *,
    flag: str,
    start_idx: int,
) -> tuple[List[Dict[str, object]], float, int]:
    """Evaluate ``tasks`` returning rows, EM and token count.

    Parameters
    ----------
    tasks:
        List of task dictionaries.
    modules:
        Active memory modules for smoke coverage.
    flag:
        Value for the ``flags`` column in ``metrics.csv``.
    start_idx:
        Starting index for row numbering.
    """

    rows: List[Dict[str, object]] = []
    correct = 0
    total_tokens = 0
    for off, item in enumerate(tasks):
        epi_feats = None
        if "episodic" in modules:
            store = modules["episodic"]["store"]
            traces = store.recall(np.zeros(8, dtype="float32"), 1)
            if traces:
                mem = torch.tensor([t.key for t in traces], dtype=torch.float32).unsqueeze(0)
            else:
                mem = torch.zeros(1, 1, 8)
            hidden_t = torch.zeros(1, 1, 8)
            epi_feats = modules["episodic"]["adapter"](hidden_t, mem).detach().numpy()[0]
        if "relational" in modules:
            modules["relational"]["kg"].retrieve(np.zeros(8, dtype="float32"))
            modules["relational"]["adapter"](
                np.zeros(8, dtype="float32"),
                np.zeros((1, 8), dtype="float32"),
                epi_feats,
            )
        if "spatial" in modules:
            modules["spatial"]["map"].plan("a", "b")
            plans = torch.zeros(1, 1, 8)
            modules["spatial"]["adapter"](torch.zeros(1, 1, 8), plans)

        prompt = str(item["prompt"])
        answer = str(item["answer"])
        pred = str(item.get("pred", answer))

        is_correct = int(pred.strip().lower() == answer.strip().lower())
        correct += is_correct
        total_tokens += len(prompt.split()) + len(answer.split())
        rows.append(
            {
                "idx": start_idx + off,
                "prompt": prompt,
                "answer": answer,
                "pred": pred,
                "correct": is_correct,
                "latency_ms": 0.0,
                "flags": flag,
            }
        )

    em = correct / len(tasks) if tasks else 0.0
    return rows, em, total_tokens


class _BatchMix(SimpleNamespace):
    episodic: float = 1.0
    semantic: float = 0.0
    fresh: float = 0.0  # kept: scheduler infers remainder; see tests/test_replay_scheduler.py


def _run_replay_once(modules: Dict[str, Dict[str, object]]) -> None:
    """Run a single dummy replay cycle using scheduler and worker."""

    store = modules.get("episodic", {}).get("store", EpisodicStore(1))
    kg = modules.get("relational", {}).get("kg", KnowledgeGraph())
    scheduler = ReplayScheduler(store, kg, batch_mix=_BatchMix())
    try:
        scheduler.add_trace("t", np.zeros(store.dim, dtype=np.float32), score=1.0)
    except Exception:
        pass
    model = torch.nn.Linear(store.dim, store.dim)
    worker = ConsolidationWorker(
        scheduler,
        model,
        episodic_adapter=modules.get("episodic", {}).get("adapter"),
        relational_adapter=modules.get("relational", {}).get("adapter"),
        spatial_adapter=modules.get("spatial", {}).get("adapter"),
        batch_size=1,
    )
    worker.start()
    time.sleep(0.05)
    worker.stop()
    worker.join(timeout=1)


def _memory_usage(modules: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Return memory usage logs from modules."""

    mem: Dict[str, object] = {}
    if "episodic" in modules:
        mem["episodic"] = modules["episodic"]["store"]._log
    if "relational" in modules:
        mem["relational"] = modules["relational"]["kg"]._log
    if "spatial" in modules:
        mem["spatial"] = modules["spatial"]["map"]._log
    return mem


def run_suite(
    cfg: DictConfig,
) -> tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    """Execute the evaluation logic and return rows and metrics."""

    tasks = generate_tasks(cfg.suite, cfg.n, cfg.seed)
    flat_ablate = _flatten_ablate(cfg.get("ablate"))
    modules = _init_modules(cfg.get("memory"), flat_ablate)

    rows, em, total_tokens = _eval_tasks(tasks, modules, flag="pre_replay", start_idx=0)

    post_cycles = int(cfg.get("post_replay_cycles", 0))
    post_metrics: Dict[str, Dict[str, object]] = {}
    if post_cycles > 0:
        start = len(tasks)
        for cycle in range(1, post_cycles + 1):
            _run_replay_once(modules)
            cycle_rows, cycle_em, cycle_tokens = _eval_tasks(
                tasks, modules, flag=f"post_replay_{cycle}", start_idx=start
            )
            rows.extend(cycle_rows)
            post_metrics[str(cycle)] = {
                cfg.suite: {"em": cycle_em},
                "compute": {"tokens": cycle_tokens},
            }
            total_tokens += cycle_tokens
            start += len(tasks)

    mem_usage = _memory_usage(modules)
    metrics = {
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "metrics": {cfg.suite: {"em": em}, "compute": {"tokens": total_tokens}},
        "memory": mem_usage,
    }
    if post_metrics:
        metrics["post_replay"] = post_metrics
    return rows, metrics, flat_ablate


def write_outputs(
    outdir: Path,
    rows: List[Dict[str, object]],
    metrics: Dict[str, object],
    flat_ablate: Dict[str, object],
    cfg: DictConfig,
) -> None:
    """Write metrics and metadata to ``outdir``."""

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f)

    mem_obj = cfg.get("memory")
    mem_dict = (
        OmegaConf.to_container(mem_obj, resolve=True) if isinstance(mem_obj, DictConfig) else {}
    )
    rel_gate = mem_dict.get("relational", {}).get("gate")
    spat_gate = mem_dict.get("spatial", {}).get("gate")
    gating_enabled = bool(
        (rel_gate or {}).get("enabled", False) or (spat_gate or {}).get("enabled", False)
    )

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
                "gating_enabled",
            ],
        )
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["gating_enabled"] = gating_enabled
            writer.writerow(row)

    cfg_hash = _config_hash(cfg)
    meta = {
        "git_sha": _git_sha(),
        "model": cfg.get("model", "mock"),
        "config_hash": cfg_hash,
        "ablate": flat_ablate,
        "seed": cfg.seed,
        "gating_enabled": gating_enabled,
    }
    config_meta: Dict[str, Dict[str, object]] = {}
    if rel_gate is not None:
        config_meta.setdefault("relational", {})["gate"] = rel_gate
    if spat_gate is not None:
        config_meta.setdefault("spatial", {})["gate"] = spat_gate
    if config_meta:
        meta["config"] = config_meta

    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)


def evaluate(cfg: DictConfig, outdir: Path) -> None:
    """Run evaluation for ``cfg.suite`` and write metrics to ``outdir``."""

    rows, metrics, flat_ablate = run_suite(cfg)
    write_outputs(outdir, rows, metrics, flat_ablate, cfg)


def evaluate_matrix(cfg: DictConfig, root_outdir: Path) -> None:
    """Run evaluation over a grid of suites, dataset sizes and seeds.

    Parameters
    ----------
    cfg:
        Hydra configuration containing ``suites``, ``n_values`` and ``seeds``
        lists in addition to the usual evaluation options.
    root_outdir:
        Directory under which per-run results will be written.  Subdirectories
        of the form ``<suite>/n<samples>_seed<seed>`` are created for each
        combination.
    """

    suites = cfg.get("suites", ["episodic", "semantic", "spatial"])
    n_values = cfg.get("n_values", [50, 200, 1000])
    seeds = cfg.get("seeds", [1337, 2025, 4242])
    base_cfg = OmegaConf.to_container(cfg, resolve=True)
    for suite in suites:
        for n in n_values:
            for seed in seeds:
                run_cfg = OmegaConf.create(base_cfg)
                run_cfg.suite = suite
                run_cfg.n = int(n)
                run_cfg.seed = int(seed)
                outdir = root_outdir / suite / f"n{n}_seed{seed}"
                mem = run_cfg.get("memory")
                if isinstance(mem, DictConfig):
                    rel_gate = (
                        mem.relational.gate
                        if mem.get("relational") and mem.relational.get("gate")
                        else None
                    )
                    spat_gate = (
                        mem.spatial.gate if mem.get("spatial") and mem.spatial.get("gate") else None
                    )
                else:
                    rel_gate = spat_gate = None
                if rel_gate or spat_gate:
                    for enabled in [True, False]:
                        gate_cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
                        if rel_gate:
                            OmegaConf.update(
                                gate_cfg, "memory.relational.gate.enabled", enabled, merge=True
                            )
                        if spat_gate:
                            OmegaConf.update(
                                gate_cfg, "memory.spatial.gate.enabled", enabled, merge=True
                            )
                        gate_dir = outdir / ("gate_on" if enabled else "gate_off")
                        evaluate(gate_cfg, gate_dir)
                else:
                    evaluate(run_cfg, outdir)


@hydra.main(version_base=None, config_path="../configs/eval", config_name="default")
def main(cfg: DictConfig) -> None:
    """CLI entry point for the evaluation harness."""

    cfg.preset = cfg.get("preset", "baselines/core")
    cfg.seed = cfg.get("seed", 0)
    cfg.n = cfg.get("n", 5)
    if cfg.get("dry_run"):
        cfg.n = min(cfg.n, 5)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    outdir: Optional[str] = cfg.get("outdir")
    preset_path = Path(str(cfg.preset))
    if cfg.get("run_matrix"):
        if outdir is not None:
            root_outdir = Path(to_absolute_path(outdir))
        else:
            if preset_path.parts and preset_path.parts[0] == "baselines":
                root_outdir = Path("runs") / date / preset_path.parts[0] / preset_path.name
            else:
                root_outdir = Path("runs") / date / preset_path.name
        evaluate_matrix(cfg, root_outdir)
    else:
        if outdir is not None:
            outdir_path = Path(to_absolute_path(outdir))
        else:
            if preset_path.parts and preset_path.parts[0] == "baselines":
                outdir_path = (
                    Path("runs") / date / preset_path.parts[0] / preset_path.name / cfg.suite
                )
            else:
                outdir_path = Path("runs") / date / preset_path.name / cfg.suite
        evaluate(cfg, outdir_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
