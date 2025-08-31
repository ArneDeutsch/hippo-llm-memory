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
from hippo_mem.common.telemetry import gate_registry, registry, set_strict_telemetry
from hippo_mem.episodic.retrieval import episodic_retrieve_and_pack
from hippo_mem.eval.score import em_norm, em_raw, f1, spatial_kpis
from hippo_mem.relational.retrieval import relational_retrieve_and_pack
from hippo_mem.spatial.retrieval import spatial_retrieve_and_pack
from hippo_mem.utils.stores import assert_store_exists

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

FORMAT_VIOL_RE = re.compile(r"\n|\.$")


def _date_str(value: object | None) -> str:
    """Return a normalized date string.

    ``value`` may be ``None``, a numeric timestamp, or a string with an optional
    ``_HHMM`` suffix. The function preserves underscores if provided and inserts
    one for 12+ digit numeric values so ``202508290841`` becomes
    ``20250829_0841``.
    """

    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    date = str(value)
    if "_" not in date and date.isdigit() and len(date) > 8:
        return f"{date[:8]}_{date[8:]}"
    return date


def _apply_model_defaults(cfg: DictConfig) -> DictConfig:
    """Populate model-related fields on ``cfg`` if missing."""
    with open_dict(cfg):
        # Allow ``task`` as a CLI alias for ``suite``.
        task = cfg.get("task")
        suite = cfg.get("suite")
        if task and not suite:
            cfg.suite = task
        elif suite and not task:
            cfg.task = suite

        cfg.n = cfg.get("n", 5)
        cfg.seed = cfg.get("seed", 0)
        m = cfg.get("model") or os.environ.get("MODEL") or os.environ.get("HF_MODEL_PATH")
        if not m:
            raise ValueError("cfg.model is missing. Pass --model or set $MODEL.")
        cfg.model = m
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
        # Normalize replay cycles key from nested config
        rc = cfg.get("replay_cycles")
        if rc in (None, 0):
            nested = cfg.get("replay") or {}
            try:
                cfg.replay_cycles = int(nested.get("cycles", 0))
            except Exception:
                cfg.replay_cycles = 0
    _merge_memory_overrides(cfg)
    return cfg


def _merge_memory_overrides(cfg: DictConfig) -> None:
    """Fold top-level memory blocks into ``cfg.memory``.

    CLI overrides may specify ``episodic.*`` or ``relational.*`` directly for
    convenience.  Hydra requires these keys to exist in the schema, so
    ``configs/eval/default.yaml`` defines them as empty dictionaries.  This
    helper merges such top-level sections into ``cfg.memory`` and removes the
    shortcuts so downstream code only sees ``cfg.memory``.
    """

    if not isinstance(cfg, DictConfig):
        return
    mem = cfg.get("memory")
    is_memory_preset = str(cfg.get("preset", "")).split("/", 1)[0] == "memory"
    with open_dict(cfg):
        if not (is_memory_preset or mem not in (None, {})):
            for name in ("episodic", "relational", "spatial"):
                if name in cfg:
                    cfg.pop(name)
            return
        if mem is None:
            cfg.memory = {}
        mem = cfg.memory
        for name in ("episodic", "relational", "spatial"):
            if name in cfg:
                block = cfg.pop(name)
                if block is None:
                    continue
                with open_dict(mem):
                    mem[name] = OmegaConf.merge(mem.get(name, {}), block)


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
    model: str = ""
    max_new_tokens: int = 32
    replay_cycles: int = 0
    use_chat_template: bool = False
    system_prompt: str | None = None
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    primary_em: str = "norm"
    dataset_profile: str | None = None


def _dataset_path(suite: str, n: int, seed: int, profile: str | None = None) -> Path:
    """Return path to a JSONL dataset for ``suite`` covering ``n`` items.

    Parameters
    ----------
    suite:
        Dataset suite name (e.g., ``episodic``).
    n:
        Number of requested items; the next canonical size >= ``n`` is used.
    seed:
        RNG seed component of the dataset filename.
    profile:
        Optional difficulty profile (``base`` or ``hard``). When provided the
        function first searches for files prefixed with ``{suite}_{profile}_`` or
        in a ``{suite}_{profile}/`` subdirectory before falling back to the base
        lookup logic.
    """

    sizes = [50, 200, 1000]
    size = next((s for s in sizes if n <= s), sizes[-1])

    candidates: List[Path] = []
    if profile and profile != "base":
        candidates.append(Path("data") / f"{suite}_{profile}_{size}_{seed}.jsonl")
        candidates.append(Path("data") / f"{suite}_{profile}" / f"{size}_{seed}.jsonl")
    candidates.append(Path("data") / f"{suite}_{size}_{seed}.jsonl")
    candidates.append(Path("data") / suite / f"{size}_{seed}.jsonl")
    for path in candidates:
        if path.exists():
            return path

    patterns = []
    if profile and profile != "base":
        patterns.append(f"{suite}_{profile}_*_{size}_{seed}.jsonl")
        patterns.append(
            f"{suite}_{profile}/{profile}_*_{size}_{seed}.jsonl"
        )  # pragma: no cover - rare
    patterns.append(f"{suite}_*_{size}_{seed}.jsonl")

    for pattern in patterns:
        matches = sorted(Path("data").glob(pattern))
        if matches:
            return matches[0]

    subdir = Path("data") / suite
    if subdir.exists():
        matches = sorted(subdir.glob(f"*_{size}_{seed}.jsonl"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "Dataset not found; run scripts/make_datasets.py or check suite name",
    )


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
    suite: str | None = None,
    retrieval_enabled: bool = True,
    long_context_enabled: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, float], int, int, float]:
    """Generate predictions and diagnostics for ``tasks``."""

    registry.reset()
    gate_registry.reset()
    rows: List[Dict[str, object]] = []
    emr_total = emn_total = f1_total = 0.0
    overlong_total = fmt_total = refusal_total = 0
    input_tokens = gen_tokens = 0
    latencies: List[float] = []
    task_list = list(tasks)
    total = len(task_list)
    progress_interval = max(1, total // 10) if total else 1
    t0 = time.perf_counter()
    for idx, item in enumerate(task_list, 1):
        item_t0 = time.perf_counter()
        if retrieval_enabled and modules:
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

        prompt = item.prompt
        if long_context_enabled and not retrieval_enabled:
            prompt = f"{prompt} [CTX]"
        inputs = encode_prompt(
            tokenizer,
            prompt,
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
        pred_len = len(pred.split())
        gold_len = len(item.answer.split())
        overlong = int(pred_len > gold_len)
        fmt = int(bool(FORMAT_VIOL_RE.search(pred)))
        em_r = em_raw(pred, item.answer) if compute_metrics else None
        em_n = em_norm(pred, item.answer) if compute_metrics else None
        f1_val = f1(pred, item.answer) if compute_metrics else None
        if compute_metrics:
            emr_total += em_r or 0
            emn_total += em_n or 0
            f1_total += f1_val or 0.0
            overlong_total += overlong
            fmt_total += fmt
            refusal_total += int(bool(REFUSAL_RE.search(pred)))
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
                "em_raw": em_r,
                "em_norm": em_n,
                "f1": f1_val,
                "pred_len": pred_len,
                "gold_len": gold_len,
                "overlong": overlong,
                "format_violation": fmt,
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
    metrics = (
        {
            "em_raw": emr_total / n if n else 0.0,
            "em_norm": emn_total / n if n else 0.0,
            "f1": f1_total / n if n else 0.0,
            "refusal_rate": refusal_total / n if n else 0.0,
            "overlong": overlong_total,
            "format_violation": fmt_total,
        }
        if compute_metrics
        else {}
    )
    if compute_metrics and suite == "spatial":
        metrics.update(spatial_kpis(task_list, rows))
    return rows, metrics, input_tokens, gen_tokens, elapsed


def _run_replay(
    cfg: DictConfig, modules: Dict[str, Dict[str, object]], tasks: Iterable[Task]
) -> int:
    """Replay loop that writes answers to the episodic store using a gate."""

    if "episodic" not in modules:
        return 0
    store = modules["episodic"]["store"]
    count = 0
    for idx, task in enumerate(tasks):
        key = np.full(8, idx, dtype="float32")
        # Deterministically insert all tasks during test runs.
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

    tasks = _load_tasks(_dataset_path(cfg.suite, cfg.n, cfg.seed, cfg.dataset_profile), cfg.n)
    flat_ablate = _flatten_ablate(base_cfg.get("ablate"))
    modules = _init_modules(base_cfg.get("memory"), flat_ablate)

    model_id = (str(base_cfg.model) or "").strip()
    if not model_id:
        raise ValueError("cfg.model is empty. Pass --model or set $MODEL.")
    p = Path(model_id)
    if p.exists() and p.is_dir():
        if not (p / "config.json").exists():
            raise ValueError(
                f"Model path '{p}' exists but is not a Hugging Face model dir (missing config.json). "
                "Did you accidentally pass the repository root? Set --model correctly."
            )

    model_path = to_absolute_path(model_id)
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

    retrieval_enabled = bool(base_cfg.get("retrieval", {}).get("enabled", False))
    long_ctx_enabled = bool(base_cfg.get("long_context", {}).get("enabled", False))

    rows, metrics, in_tokens, gen_tokens, elapsed = _evaluate(
        tasks,
        modules,
        tokenizer,
        model,
        int(base_cfg.max_new_tokens),
        use_chat_template=cfg.use_chat_template,
        system_prompt=cfg.system_prompt,
        suite=cfg.suite,
        retrieval_enabled=retrieval_enabled,
        long_context_enabled=long_ctx_enabled,
    )
    metrics["em"] = (
        metrics.get("em_norm", 0.0) if cfg.primary_em == "norm" else metrics.get("em_raw", 0.0)
    )
    lat_mean = sum(r["latency_ms"] for r in rows) / max(1, len(rows))

    replay_samples = 0
    for _ in range(int(cfg.replay_cycles)):
        replay_samples += _run_replay(base_cfg, modules, tasks)

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
    is_test = str(cfg.get("mode")) in ("test", "teach")

    # Metrics JSON - follow schema used by report.py
    suite_metrics: Dict[str, float] = {
        "pre_em": pre_metrics.get("em", 0.0),
        "pre_em_raw": pre_metrics.get("em_raw", 0.0),
        "pre_em_norm": pre_metrics.get("em_norm", 0.0),
        "pre_f1": pre_metrics.get("f1", 0.0),
        "pre_refusal_rate": pre_metrics.get("refusal_rate", 0.0),
    }
    for k in ("success_rate", "suboptimality_ratio", "steps_to_goal"):
        if k in pre_metrics:
            suite_metrics[f"pre_{k}"] = pre_metrics[k]
    diagnostics: Dict[str, int] = {
        "pre_overlong": pre_metrics.get("overlong", 0),
        "pre_format_violation": pre_metrics.get("format_violation", 0),
    }
    if post_metrics is not None:
        suite_metrics.update(
            {
                "post_em": post_metrics.get("em", 0.0),
                "post_em_raw": post_metrics.get("em_raw", 0.0),
                "post_em_norm": post_metrics.get("em_norm", 0.0),
                "post_f1": post_metrics.get("f1", 0.0),
                "post_refusal_rate": post_metrics.get("refusal_rate", 0.0),
            }
        )
        for k in ("success_rate", "suboptimality_ratio", "steps_to_goal"):
            if k in post_metrics:
                suite_metrics[f"post_{k}"] = post_metrics[k]
        diagnostics.update(
            {
                "post_overlong": post_metrics.get("overlong", 0),
                "post_format_violation": post_metrics.get("format_violation", 0),
            }
        )
        for key, pre_val in pre_metrics.items():
            post_val = post_metrics.get(key)
            if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)):
                suite_metrics[f"delta_{key}"] = post_val - pre_val
    metrics: Dict[str, object] = {
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "metrics": {cfg.suite: suite_metrics},
        "diagnostics": {cfg.suite: diagnostics},
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

    if is_test:
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
            f"retrieval.{m}.{k}": v
            for m, snap in metrics.get("retrieval", {}).items()
            for k, v in snap.items()
        }
        gate_fields = {
            f"gates.{m}.{k}": v
            for m, snap in metrics.get("gates", {}).items()
            for k, v in snap.items()
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
            "em_raw",
            "em_norm",
            "f1",
            "pred_len",
            "gold_len",
            "overlong",
            "format_violation",
            "latency_ms",
            "success",
            "steps_pred",
            "steps_opt",
            "suboptimality",
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
        "date": cfg.get("date"),
        "git_sha": _git_sha(),
        "model": model_meta,
        "config_hash": _config_hash(cfg),
        "ablate": flat_ablate,
        "seed": cfg.seed,
        "replay_cycles": cfg.get("replay_cycles", cfg.get("replay", {}).get("cycles", 0)),
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

    dataset = _dataset_path(cfg.suite, cfg.n, cfg.seed, cfg.dataset_profile)
    tasks = _load_tasks(dataset, cfg.n)
    flat_ablate = _flatten_ablate(cfg.get("ablate"))
    mem_cfg = cfg.get("memory")
    if isinstance(mem_cfg, DictConfig):
        with open_dict(mem_cfg):
            epi_cfg = mem_cfg.get("episodic")
            if isinstance(epi_cfg, DictConfig):
                if "episodic.use_gate" in flat_ablate:
                    gate_cfg = epi_cfg.get("gate")
                    if isinstance(gate_cfg, DictConfig):
                        with open_dict(gate_cfg):
                            gate_cfg.enabled = bool(flat_ablate["episodic.use_gate"])
                if "episodic.use_completion" in flat_ablate:
                    epi_cfg["use_completion"] = bool(flat_ablate["episodic.use_completion"])
            rel_cfg = mem_cfg.get("relational")
            if isinstance(rel_cfg, DictConfig) and "relational.gate.enabled" in flat_ablate:
                gate_cfg = rel_cfg.get("gate")
                if isinstance(gate_cfg, DictConfig):
                    with open_dict(gate_cfg):
                        gate_cfg.enabled = bool(flat_ablate["relational.gate.enabled"])
            spat_cfg = mem_cfg.get("spatial")
            if isinstance(spat_cfg, DictConfig) and "spatial.gate.enabled" in flat_ablate:
                gate_cfg = spat_cfg.get("gate")
                if isinstance(gate_cfg, DictConfig):
                    with open_dict(gate_cfg):
                        gate_cfg.enabled = bool(flat_ablate["spatial.gate.enabled"])
    modules = _init_modules(mem_cfg, flat_ablate)
    if cfg.memory_off:
        modules = {}
    elif cfg.mode in ("test", "replay") and cfg.store_dir and cfg.session_id:
        session_dir = Path(to_absolute_path(str(cfg.store_dir)))
        sid = str(cfg.session_id)
        assert_store_exists(str(session_dir.parent), sid, session_dir.name)
        if "episodic" in modules:
            modules["episodic"]["store"].load(str(session_dir), sid)
        if "relational" in modules:
            kg_file = session_dir / sid / "kg.jsonl"
            rel_file = session_dir / sid / "relational.jsonl"
            if kg_file.exists() or rel_file.exists():
                modules["relational"]["kg"].load(str(session_dir), sid)
        if "spatial" in modules:
            spat_file = session_dir / sid / "spatial.jsonl"
            if spat_file.exists():
                modules["spatial"]["map"].load(str(session_dir), sid)

    retrieval_enabled = bool(cfg.get("retrieval", {}).get("enabled", False))
    long_ctx_enabled = bool(cfg.get("long_context", {}).get("enabled", False))

    model_id = (str(cfg.model) or "").strip()
    if not model_id:
        raise ValueError("cfg.model is empty. Pass --model or set $MODEL.")
    p = Path(model_id)
    if p.exists() and p.is_dir():
        if not (p / "config.json").exists():
            raise ValueError(
                f"Model path '{p}' exists but is not a Hugging Face model dir (missing config.json). "
                "Did you accidentally pass the repository root? Set --model correctly."
            )
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

    replay_samples = 0

    if cfg.mode == "teach":
        pre_rows, _, _, _, _ = _evaluate(
            tasks,
            modules,
            tokenizer,
            model,
            int(cfg.max_new_tokens),
            use_chat_template=cfg.use_chat_template,
            system_prompt=cfg.system_prompt,
            compute_metrics=False,
            suite=cfg.suite,
            retrieval_enabled=retrieval_enabled,
            long_context_enabled=long_ctx_enabled,
        )
        if cfg.persist and cfg.store_dir and cfg.session_id:
            session_dir = Path(to_absolute_path(str(cfg.store_dir)))
            if "episodic" in modules:
                modules["episodic"]["store"].save(str(session_dir), str(cfg.session_id))
            if "relational" in modules:
                modules["relational"]["kg"].save(str(session_dir), str(cfg.session_id))
            if "spatial" in modules:
                modules["spatial"]["map"].save(str(session_dir), str(cfg.session_id))
        store_sizes = _store_sizes(modules)
        _write_outputs(
            outdir,
            pre_rows,
            {},
            None,
            None,
            cfg,
            flat_ablate,
            None,
            replay_samples=0,
            store_sizes=store_sizes,
        )
        return

    if cfg.mode == "replay":
        for _ in range(int(cfg.replay_cycles)):
            replay_samples += _run_replay(cfg, modules, tasks)
        if cfg.persist and cfg.store_dir and cfg.session_id:
            session_dir = Path(to_absolute_path(str(cfg.store_dir)))
            if "episodic" in modules:
                modules["episodic"]["store"].save(str(session_dir), str(cfg.session_id))
            if "relational" in modules:
                modules["relational"]["kg"].save(str(session_dir), str(cfg.session_id))
            if "spatial" in modules:
                modules["spatial"]["map"].save(str(session_dir), str(cfg.session_id))

        # Load pre-metrics if they exist so we can compute deltas.
        pre_metrics: Dict[str, float] = {}
        metrics_path = outdir / "metrics.json"
        if not metrics_path.exists():
            msg = f"missing {metrics_path}; run pre phase in the same outdir before replay"
            raise FileNotFoundError(msg)
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        suite_metrics = data.get("metrics", {}).get(cfg.suite, {})
        diagnostics = data.get("diagnostics", {}).get(cfg.suite, {})
        for key in (
            "em",
            "em_raw",
            "em_norm",
            "f1",
            "refusal_rate",
            "success_rate",
            "suboptimality_ratio",
            "steps_to_goal",
        ):
            val = suite_metrics.get(f"pre_{key}")
            if val is not None:
                pre_metrics[key] = float(val)
        for key in ("overlong", "format_violation"):
            val = diagnostics.get(f"pre_{key}")
            if val is not None:
                pre_metrics[key] = float(val)

        registry.reset()
        gate_registry.reset()
        post_rows, post_metrics, in_tok, gen_tok, elapsed = _evaluate(
            tasks,
            modules,
            tokenizer,
            model,
            int(cfg.max_new_tokens),
            use_chat_template=cfg.use_chat_template,
            system_prompt=cfg.system_prompt,
            compute_metrics=True,
            suite=cfg.suite,
            retrieval_enabled=retrieval_enabled,
            long_context_enabled=long_ctx_enabled,
        )
        post_metrics["em"] = (
            post_metrics.get("em_norm", 0.0)
            if cfg.primary_em == "norm"
            else post_metrics.get("em_raw", 0.0)
        )
        total_tokens = in_tok + gen_tok
        compute = {
            "input_tokens": in_tok,
            "generated_tokens": gen_tok,
            "total_tokens": total_tokens,
            "time_ms_per_100": 100 * elapsed * 1000 / max(1, total_tokens),
            "rss_mb": _rss_mb(),
            "latency_ms_mean": sum(r["latency_ms"] for r in post_rows) / max(1, len(post_rows)),
        }
        store_sizes = _store_sizes(modules)
        _write_outputs(
            outdir,
            [],
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
        _enforce_guardrails(
            cfg, pre_metrics, post_metrics, retrieval_snaps, has_memory=bool(modules)
        )
        return

    pre_rows, pre_metrics, pre_in_tokens, pre_gen_tokens, pre_time = _evaluate(
        tasks,
        modules,
        tokenizer,
        model,
        int(cfg.max_new_tokens),
        use_chat_template=cfg.use_chat_template,
        system_prompt=cfg.system_prompt,
        compute_metrics=True,
        suite=cfg.suite,
        retrieval_enabled=retrieval_enabled,
        long_context_enabled=long_ctx_enabled,
    )
    pre_metrics["em"] = (
        pre_metrics.get("em_norm", 0.0)
        if cfg.primary_em == "norm"
        else pre_metrics.get("em_raw", 0.0)
    )
    latencies = [row["latency_ms"] for row in pre_rows]
    total_in_tokens = pre_in_tokens
    total_gen_tokens = pre_gen_tokens
    total_time = pre_time

    total_tokens = total_in_tokens + total_gen_tokens
    compute = {
        "input_tokens": total_in_tokens,
        "generated_tokens": total_gen_tokens,
        "total_tokens": total_tokens,
        "time_ms_per_100": 100 * total_time * 1000 / max(1, total_tokens),
        "rss_mb": _rss_mb(),
        "latency_ms_mean": sum(latencies) / max(1, len(latencies)),
    }
    for _ in range(int(cfg.replay_cycles)):
        replay_samples += _run_replay(cfg, modules, tasks)
    store_sizes = _store_sizes(modules)
    _write_outputs(
        outdir,
        pre_rows,
        pre_metrics,
        None,
        None,
        cfg,
        flat_ablate,
        compute,
        replay_samples=replay_samples,
        store_sizes=store_sizes,
    )

    retrieval_snaps = registry.all_snapshots()
    _enforce_guardrails(cfg, pre_metrics, None, retrieval_snaps, has_memory=bool(modules))


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

    # Normalise and freeze the date early so repeated accesses do not drift.
    with open_dict(cfg):
        run_date = _date_str(cfg.get("date"))
        cfg.date = run_date
        set_strict_telemetry(cfg.get("strict_telemetry", False))

    base_cfg = cfg
    outdir_cfg = cfg.get("outdir")
    if cfg.get("run_matrix"):
        presets = cfg.get("presets")
        if presets:
            if outdir_cfg is not None:
                base_outdir = Path(to_absolute_path(str(outdir_cfg)))
            else:
                base_outdir = Path("runs") / run_date
            for preset in presets:
                run_cfg = OmegaConf.merge(base_cfg, {"preset": preset})
                run_cfg = _load_preset(run_cfg)
                run_cfg = _apply_model_defaults(run_cfg)
                with open_dict(run_cfg):
                    run_cfg.date = run_date
                preset_outdir = base_outdir / Path(preset)
                evaluate_matrix(run_cfg, preset_outdir)
        else:
            cfg = _load_preset(cfg)
            cfg = _apply_model_defaults(cfg)
            with open_dict(cfg):
                cfg.date = run_date
            if outdir_cfg is not None:
                root_outdir = Path(to_absolute_path(str(outdir_cfg)))
            else:
                preset_path = Path(str(cfg.preset))
                if preset_path.parts and preset_path.parts[0] == "baselines":
                    root_outdir = Path("runs") / run_date / preset_path.parts[0] / preset_path.name
                else:
                    root_outdir = Path("runs") / run_date / preset_path.name
            evaluate_matrix(cfg, root_outdir)
    else:
        cfg = _load_preset(cfg)
        cfg = _apply_model_defaults(cfg)
        with open_dict(cfg):
            cfg.date = run_date
        if outdir_cfg is not None:
            outdir = Path(to_absolute_path(str(outdir_cfg)))
        else:
            preset_path = Path(str(cfg.preset))
            if preset_path.parts and preset_path.parts[0] == "baselines":
                outdir = (
                    Path("runs") / run_date / preset_path.parts[0] / preset_path.name / cfg.suite
                )
            else:
                outdir = Path("runs") / run_date / preset_path.name / cfg.suite
        evaluate(cfg, outdir)
