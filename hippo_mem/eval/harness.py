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
import itertools
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.common.telemetry import gate_registry, registry
from hippo_mem.episodic.retrieval import episodic_retrieve_and_pack
from hippo_mem.relational.retrieval import relational_retrieve_and_pack
from hippo_mem.spatial.retrieval import spatial_retrieve_and_pack

from .bench import _config_hash, _flatten_ablate, _git_sha, _init_modules
from .encode import encode_prompt
from .models import load_model_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REFUSAL_RE = re.compile(
    r"(i\s+can't|i\s+cannot|i\s+won't|i\s+will\s+not|as\s+an\s+ai|"
    r"as\s+an\s+language\s+model|i\s+am\s+unable|i\s+do\s+not\s+have)",
    re.IGNORECASE,
)


def _apply_model_defaults(cfg: DictConfig) -> DictConfig:
    """Populate model-related fields on ``cfg`` if missing."""
    # Allow ``task`` as a CLI alias for ``suite``.
    task = cfg.get("task")
    suite = cfg.get("suite")
    if task and not suite:
        cfg.suite = task
    elif suite and not task:
        cfg.task = suite

    cfg.n = cfg.get("n", 5)
    cfg.seed = cfg.get("seed", 0)
    cfg.model = cfg.get("model", "models/tiny-gpt2")
    cfg.max_new_tokens = cfg.get("max_new_tokens")
    cfg.mode = cfg.get("mode", "test")
    cfg.store_dir = cfg.get("store_dir")
    cfg.session_id = cfg.get("session_id")
    cfg.persist = cfg.get("persist", False)
    cfg.memory_off = cfg.get("memory_off", False)
    model_cfg = load_model_config(str(cfg.model))
    if cfg.get("use_chat_template") is None:
        cfg.use_chat_template = model_cfg.get("use_chat_template", False)
    if cfg.get("system_prompt") is None:
        cfg.system_prompt = model_cfg.get("system_prompt")
    if cfg.get("pad_token_id") is None:
        cfg.pad_token_id = model_cfg.get("pad_token_id")
    if cfg.get("eos_token_id") is None:
        cfg.eos_token_id = model_cfg.get("eos_token_id")
    if cfg.max_new_tokens is None:
        cfg.max_new_tokens = model_cfg.get("max_new_tokens", 32)
    force_chat = cfg.get("force_chat")
    force_no_chat = cfg.get("force_no_chat")
    if force_chat and force_no_chat:
        raise ValueError("force_chat and force_no_chat are mutually exclusive")
    if force_chat:
        cfg.use_chat_template = True
    elif force_no_chat:
        cfg.use_chat_template = False
    if cfg.get("dry_run"):
        cfg.n = min(cfg.n, 5)
    return cfg


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
    use_chat_template: bool = False
    system_prompt: str | None = None
    pad_token_id: int | None = None
    eos_token_id: int | None = None


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


def _rss_mb() -> float:
    """Return resident set size of current process in megabytes."""

    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover - psutil may be missing
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss /= 1024
        return rss / 1024


def _evaluate(
    tasks: Iterable[Task],
    modules: Dict[str, Dict[str, object]],
    tokenizer,
    model,
    max_new_tokens: int,
    *,
    use_chat_template: bool,
    system_prompt: str | None,
    compute_metrics: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, float], int, float]:
    """Generate predictions and optionally compute metrics for ``tasks``.

    Returns the per-item rows, metric averages (empty if ``compute_metrics`` is
    ``False``), input and generated token counts, and elapsed seconds.
    """

    rows: List[Dict[str, object]] = []
    correct = 0
    f1_total = 0.0
    input_tokens = 0
    gen_tokens = 0
    latencies: List[float] = []
    task_list = list(tasks)
    total = len(task_list)
    progress_interval = max(1, total // 10) if total else 1
    t0 = time.perf_counter()
    for idx, item in enumerate(task_list, 1):
        item_t0 = time.perf_counter()
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

        inputs = encode_prompt(
            tokenizer,
            item.prompt,
            model.device,
            use_chat_template=use_chat_template,
            system_prompt=(
                system_prompt
                or "Answer with the exact shortest span from the prompt. No explanations."
            ),
        )
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out[:, inputs["input_ids"].shape[-1] :]
        pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
        is_correct = int(pred.strip() == item.answer)
        if compute_metrics:
            correct += is_correct
            f1_total += _f1(pred, item.answer)
        else:
            is_correct = None  # type: ignore[assignment]
        input_tokens += inputs["input_ids"].shape[-1]
        gen_tokens += gen.shape[-1]
        item_t1 = time.perf_counter()
        latency_ms = (item_t1 - item_t0) * 1000.0
        latencies.append(latency_ms)
        rows.append(
            {
                "idx": idx - 1,
                "prompt": item.prompt,
                "answer": item.answer,
                "pred": pred,
                "correct": is_correct,
                "latency_ms": latency_ms,
            }
        )
        if idx % progress_interval == 0 or idx == total:
            log.info("    processed %d/%d tasks", idx, total)
    n = len(rows)
    t1 = time.perf_counter()
    if all(lat == 0.0 for lat in latencies) and latencies:
        fallback = (t1 - t0) * 1000.0 / len(latencies)
        for row in rows:
            row["latency_ms"] = fallback
        latencies = [fallback] * len(rows)
    elapsed = t1 - t0
    refusal_count = sum(1 for r in rows if REFUSAL_RE.search(r["pred"]))
    metrics = (
        {
            "em": correct / n if n else 0.0,
            "f1": f1_total / n if n else 0.0,
            "refusal_rate": refusal_count / n if n else 0.0,
        }
        if compute_metrics
        else {}
    )
    return rows, metrics, input_tokens, gen_tokens, elapsed


def _run_replay(modules: Dict[str, Dict[str, object]], tasks: Iterable[Task]) -> int:
    """Dummy replay loop that writes answers to the episodic store."""

    if "episodic" not in modules:
        return 0
    store = modules["episodic"]["store"]
    count = 0
    for task in tasks:
        key = np.ones(8, dtype="float32")
        store.write(key, task.answer)
        count += 1
    return count


def _store_sizes(modules: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    """Return the number of items per memory store."""

    sizes: Dict[str, int] = {}
    if "episodic" in modules:
        store = modules["episodic"]["store"]
        try:
            cur = store.persistence.db.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM traces")
            sizes["episodic"] = int(cur.fetchone()[0])
        except Exception:
            sizes["episodic"] = 0
    if "relational" in modules:
        kg = modules["relational"]["kg"]
        sizes["relational"] = int(kg.graph.number_of_edges())
    if "spatial" in modules:
        g = modules["spatial"].get("map")
        sizes["spatial"] = int(g.log_status().get("writes", 0))
    return sizes


def _enforce_guardrails(
    cfg: DictConfig,
    pre_metrics: Dict[str, float],
    post_metrics: Optional[Dict[str, float]],
    retrieval_snaps: Dict[str, Dict[str, float]],
    *,
    has_memory: bool,
) -> None:
    """Raise errors when CI guardrails are violated."""

    if has_memory and not cfg.get("memory_off"):
        total = sum(int(snap.get("requests", 0)) for snap in retrieval_snaps.values())
        if total == 0:
            raise RuntimeError("retrieval.requests == 0 for memory run")

    if cfg.get("use_chat_template") and int(cfg.get("max_new_tokens", 0)) <= 16:
        rates = [pre_metrics.get("refusal_rate", 0.0)]
        if post_metrics is not None:
            rates.append(post_metrics.get("refusal_rate", 0.0))
        if any(rate > 0.5 for rate in rates):
            raise RuntimeError("refusal rate > 0.5 on span suite")


def run_suite(
    cfg: EvalConfig,
) -> tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    """Execute a suite according to ``cfg`` and return rows and metrics."""

    registry.reset()
    gate_registry.reset()

    base_cfg = OmegaConf.create(
        {
            "suite": cfg.suite,
            "n": cfg.n,
            "seed": cfg.seed,
            "model": cfg.model,
            "max_new_tokens": cfg.max_new_tokens,
            "use_chat_template": cfg.use_chat_template,
            "system_prompt": cfg.system_prompt,
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
    if cfg.pad_token_id is not None:
        tokenizer.pad_token_id = cfg.pad_token_id
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.eos_token_id is not None:
        tokenizer.eos_token_id = cfg.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if cfg.eos_token_id is not None:
        model.generation_config.eos_token_id = cfg.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows, metrics, in_tokens, gen_tokens, elapsed = _evaluate(
        tasks,
        modules,
        tokenizer,
        model,
        int(base_cfg.max_new_tokens),
        use_chat_template=cfg.use_chat_template,
        system_prompt=cfg.system_prompt,
    )
    lat_mean = sum(r["latency_ms"] for r in rows) / max(1, len(rows))

    replay_samples = 0
    for _ in range(int(cfg.replay_cycles)):
        replay_samples += _run_replay(modules, tasks)

    total_tokens = in_tokens + gen_tokens
    store_sizes = _store_sizes(modules)
    retrieval_snaps = registry.all_snapshots()
    metrics_dict = {
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "metrics": {
            cfg.suite: metrics,
            "compute": {
                "input_tokens": in_tokens,
                "generated_tokens": gen_tokens,
                "total_tokens": total_tokens,
                "time_ms_per_100": 100 * elapsed * 1000 / max(1, total_tokens),
                "rss_mb": _rss_mb(),
                "latency_ms_mean": lat_mean,
            },
        },
        "retrieval": retrieval_snaps,
        "gates": gate_registry.all_snapshots(),
        "replay": {"samples": replay_samples},
        "store": {
            "size": sum(store_sizes.values()),
            "per_memory": store_sizes,
        },
    }
    _enforce_guardrails(base_cfg, metrics, None, retrieval_snaps, has_memory=bool(modules))
    return rows, metrics_dict, flat_ablate


def _write_outputs(
    outdir: Path,
    pre_rows: List[Dict[str, object]],
    pre_metrics: Dict[str, float],
    post_rows: Optional[List[Dict[str, object]]],
    post_metrics: Optional[Dict[str, float]],
    cfg: DictConfig,
    flat_ablate: Dict[str, object],
    compute: Dict[str, float] | None = None,
    *,
    replay_samples: int = 0,
    store_sizes: Dict[str, int] | None = None,
) -> None:
    """Persist metrics and metadata."""

    outdir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON - follow schema used by report.py
    suite_metrics: Dict[str, float] = {
        "pre_em": pre_metrics.get("em", 0.0),
        "pre_f1": pre_metrics.get("f1", 0.0),
        "pre_refusal_rate": pre_metrics.get("refusal_rate", 0.0),
    }
    if post_metrics is not None:
        suite_metrics.update(
            {
                "post_em": post_metrics.get("em", 0.0),
                "post_f1": post_metrics.get("f1", 0.0),
                "post_refusal_rate": post_metrics.get("refusal_rate", 0.0),
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
        "gates": gate_registry.all_snapshots(),
        "replay": {"samples": replay_samples},
        "store": {
            "size": sum((store_sizes or {}).values()),
            "per_memory": store_sizes or {},
        },
    }
    if compute:
        metrics["metrics"]["compute"] = compute
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
    mem_obj = cfg.get("memory")
    mem_dict = (
        OmegaConf.to_container(mem_obj, resolve=True) if isinstance(mem_obj, DictConfig) else {}
    )
    epi_gate = mem_dict.get("episodic", {}).get("gate")
    rel_gate = mem_dict.get("relational", {}).get("gate")
    spat_gate = mem_dict.get("spatial", {}).get("gate")
    gating_enabled = bool(
        (epi_gate or {}).get("enabled", False)
        or (rel_gate or {}).get("enabled", False)
        or (spat_gate or {}).get("enabled", False)
    )
    retrieval_fields = {
        f"retrieval.{m}.{k}": v
        for m, snap in metrics.get("retrieval", {}).items()
        for k, v in snap.items()
    }
    gate_fields = {
        f"gates.{m}.{k}": v for m, snap in metrics.get("gates", {}).items() for k, v in snap.items()
    }
    compute_cols = [k for k in ("time_ms_per_100", "rss_mb") if compute and k in compute]
    for row in csv_rows:
        row.update(retrieval_fields)
        row.update(gate_fields)
        for col in compute_cols:
            row[col] = compute.get(col) if compute else None
        row["gating_enabled"] = gating_enabled
    fieldnames = [
        "idx",
        "prompt",
        "answer",
        "pred",
        "correct",
        "latency_ms",
        *compute_cols,
        "flags",
        "gating_enabled",
        *sorted(retrieval_fields),
        *sorted(gate_fields),
    ]
    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # Small audit sample
    sample = []
    for row in pre_rows[:10]:
        sample.append(
            {
                "id": row["idx"],
                "prompt": str(row["prompt"])[:2000],
                "answer": str(row["answer"])[:2000],
                "pred": str(row["pred"])[:2000],
            }
        )
    with (outdir / "audit_sample.jsonl").open("w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec) + "\n")

    # Metadata JSON
    model_meta = {
        "id": cfg.get("model", "mock"),
        "chat_template_used": bool(cfg.get("use_chat_template", False)),
    }
    if cfg.get("force_chat"):
        model_meta["force_chat"] = True
    if cfg.get("force_no_chat"):
        model_meta["force_no_chat"] = True

    meta = {
        "suite": cfg.suite,
        "preset": cfg.preset,
        "n": cfg.n,
        "git_sha": _git_sha(),
        "model": model_meta,
        "config_hash": _config_hash(cfg),
        "ablate": flat_ablate,
        "seed": cfg.seed,
        "replay_cycles": cfg.get("replay", {}).get("cycles", 0),
        "gating_enabled": gating_enabled,
        "mode": cfg.get("mode"),
        "store_dir": cfg.get("store_dir"),
        "session_id": cfg.get("session_id"),
        "persist": cfg.get("persist"),
        "memory_off": cfg.get("memory_off"),
    }
    config_meta: Dict[str, Dict[str, object]] = {}
    if epi_gate is not None:
        config_meta.setdefault("episodic", {})["gate"] = epi_gate
    if rel_gate is not None:
        config_meta.setdefault("relational", {})["gate"] = rel_gate
    if spat_gate is not None:
        config_meta.setdefault("spatial", {})["gate"] = spat_gate
    if config_meta:
        meta["config"] = config_meta

    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)


def _load_preset(cfg: DictConfig) -> DictConfig:
    """Merge the ``preset`` YAML into ``cfg`` if it exists."""

    preset = cfg.get("preset", "baselines/core")
    preset_path = Path(to_absolute_path("configs")) / "eval" / f"{preset}.yaml"
    if preset_path.exists():
        preset_cfg = OmegaConf.load(preset_path)
        with open_dict(cfg):
            cfg = OmegaConf.merge(preset_cfg, cfg)  # CLI overrides preset
    return cfg


def evaluate(cfg: DictConfig, outdir: Path) -> None:
    """Run a single evaluation and write outputs to ``outdir``."""

    registry.reset()
    gate_registry.reset()

    dataset = _dataset_path(cfg.suite, cfg.n, cfg.seed)
    tasks = _load_tasks(dataset, cfg.n)
    flat_ablate = _flatten_ablate(cfg.get("ablate"))
    modules = _init_modules(cfg.get("memory"), flat_ablate)
    if cfg.memory_off:
        modules = {}
    elif cfg.mode == "test" and cfg.store_dir and cfg.session_id:
        session_dir = Path(to_absolute_path(str(cfg.store_dir)))
        sid = str(cfg.session_id)
        if "episodic" in modules:
            epi_file = session_dir / sid / "episodic.jsonl"
            if epi_file.exists():
                modules["episodic"]["store"].load(str(session_dir), sid)
        if "relational" in modules:
            rel_file = session_dir / sid / "relational.jsonl"
            if rel_file.exists():
                modules["relational"]["kg"].load(str(session_dir), sid)
        if "spatial" in modules:
            spat_file = session_dir / sid / "spatial.jsonl"
            if spat_file.exists():
                modules["spatial"]["map"].load(str(session_dir), sid)

    model_id = str(cfg.model)
    abs_model_path = Path(to_absolute_path(model_id))
    model_path = abs_model_path if abs_model_path.exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if cfg.pad_token_id is not None:
        tokenizer.pad_token_id = cfg.pad_token_id
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.eos_token_id is not None:
        tokenizer.eos_token_id = cfg.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if cfg.eos_token_id is not None:
        model.generation_config.eos_token_id = cfg.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pre_rows, pre_metrics, pre_in_tokens, pre_gen_tokens, pre_time = _evaluate(
        tasks,
        modules,
        tokenizer,
        model,
        int(cfg.max_new_tokens),
        use_chat_template=cfg.use_chat_template,
        system_prompt=cfg.system_prompt,
        compute_metrics=cfg.mode != "teach",
    )
    latencies = [row["latency_ms"] for row in pre_rows]
    post_rows = post_metrics = None
    total_items = len(pre_rows)
    total_in_tokens = pre_in_tokens
    total_gen_tokens = pre_gen_tokens
    total_time = pre_time
    replay_samples = 0

    if cfg.mode == "teach":
        replay_samples += _run_replay(modules, tasks)
        if cfg.persist and cfg.store_dir and cfg.session_id:
            session_dir = Path(to_absolute_path(str(cfg.store_dir)))
            if "episodic" in modules:
                modules["episodic"]["store"].save(str(session_dir), str(cfg.session_id))
            if "relational" in modules:
                modules["relational"]["kg"].save(str(session_dir), str(cfg.session_id))
            if "spatial" in modules:
                modules["spatial"]["map"].save(str(session_dir), str(cfg.session_id))
    else:
        for _ in range(int(cfg.get("replay", {}).get("cycles", 0))):
            replay_samples += _run_replay(modules, tasks)
            post_rows, post_metrics, post_in_tokens, post_gen_tokens, post_time = _evaluate(
                tasks,
                modules,
                tokenizer,
                model,
                int(cfg.max_new_tokens),
                use_chat_template=cfg.use_chat_template,
                system_prompt=cfg.system_prompt,
            )
            latencies.extend(row["latency_ms"] for row in post_rows)
            total_items += len(post_rows)
            total_in_tokens += post_in_tokens
            total_gen_tokens += post_gen_tokens
            total_time += post_time

    total_tokens = total_in_tokens + total_gen_tokens
    compute = {
        "input_tokens": total_in_tokens,
        "generated_tokens": total_gen_tokens,
        "total_tokens": total_tokens,
        "time_ms_per_100": 100 * total_time * 1000 / max(1, total_tokens),
        "rss_mb": _rss_mb(),
        "latency_ms_mean": sum(latencies) / max(1, len(latencies)),
    }
    store_sizes = _store_sizes(modules)
    _write_outputs(
        outdir,
        pre_rows,
        pre_metrics,
        post_rows,
        post_metrics,
        cfg,
        flat_ablate,
        compute,
        replay_samples=replay_samples,
        store_sizes=store_sizes,
    )

    retrieval_snaps = registry.all_snapshots()
    _enforce_guardrails(cfg, pre_metrics, post_metrics, retrieval_snaps, has_memory=bool(modules))


def evaluate_matrix(cfg: DictConfig, root_outdir: Path) -> None:
    """Run evaluation over a grid of tasks, sizes and seeds."""

    tasks = cfg.get("tasks")
    if not tasks:
        tasks = cfg.get("suites", ["episodic", "semantic", "spatial"])
    n_values = cfg.get("n_values", [50, 200, 1000])
    seeds = cfg.get("seeds", [1337, 2025, 4242])
    combos = list(itertools.product(tasks, n_values, seeds))
    total = len(combos)
    base_cfg = OmegaConf.to_container(cfg, resolve=True)
    for idx, (task, n, seed) in enumerate(combos, 1):
        log.info("run %d/%d: task=%s n=%d seed=%d", idx, total, task, n, seed)
        run_cfg = OmegaConf.create(base_cfg)
        run_cfg.suite = task
        run_cfg.n = int(n)
        run_cfg.seed = int(seed)
        outdir = root_outdir / task / f"{n}_{seed}"
        evaluate(run_cfg, outdir)


def main(cfg: DictConfig) -> None:
    """Run evaluation based on ``cfg``."""

    base_cfg = cfg
    outdir_cfg = cfg.get("outdir")
    if cfg.get("run_matrix"):
        presets = cfg.get("presets")
        if presets:
            if outdir_cfg is not None:
                base_outdir = Path(to_absolute_path(str(outdir_cfg)))
            else:
                date = str(cfg.get("date") or datetime.now(timezone.utc).strftime("%Y%m%d"))
                base_outdir = Path("runs") / date
            for preset in presets:
                run_cfg = OmegaConf.merge(base_cfg, {"preset": preset})
                run_cfg = _load_preset(run_cfg)
                run_cfg = _apply_model_defaults(run_cfg)
                preset_outdir = base_outdir / Path(preset)
                evaluate_matrix(run_cfg, preset_outdir)
        else:
            cfg = _load_preset(cfg)
            cfg = _apply_model_defaults(cfg)
            if outdir_cfg is not None:
                root_outdir = Path(to_absolute_path(str(outdir_cfg)))
            else:
                date = str(cfg.get("date") or datetime.now(timezone.utc).strftime("%Y%m%d"))
                preset_path = Path(str(cfg.preset))
                if preset_path.parts and preset_path.parts[0] == "baselines":
                    root_outdir = Path("runs") / date / preset_path.parts[0] / preset_path.name
                else:
                    root_outdir = Path("runs") / date / preset_path.name
            evaluate_matrix(cfg, root_outdir)
    else:
        cfg = _load_preset(cfg)
        cfg = _apply_model_defaults(cfg)
        if outdir_cfg is not None:
            outdir = Path(to_absolute_path(str(outdir_cfg)))
        else:
            date = str(cfg.get("date") or datetime.now(timezone.utc).strftime("%Y%m%d"))
            preset_path = Path(str(cfg.preset))
            if preset_path.parts and preset_path.parts[0] == "baselines":
                outdir = Path("runs") / date / preset_path.parts[0] / preset_path.name / cfg.suite
            else:
                outdir = Path("runs") / date / preset_path.name / cfg.suite
        evaluate(cfg, outdir)
