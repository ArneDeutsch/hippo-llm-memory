# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
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

import itertools
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from hippo_eval.bench import _config_hash, _flatten_ablate, _git_sha, _init_modules
from hippo_eval.datasets.loaders import load_dataset
from hippo_eval.eval.adapters import (
    EpisodicEvalAdapter,
    EvalAdapter,
    RelationalEvalAdapter,
    SpatialEvalAdapter,
    enabled_adapters,
)
from hippo_eval.eval.modes import Mode, ModeStrategy, TeachStrategy, get_mode_strategy
from hippo_eval.eval.types import Task
from hippo_eval.harness.metrics import collect_metrics
from hippo_eval.metrics.scoring import (
    em_norm,
    f1,
    oracle_from_context,
    spatial_kpis,
    spatial_multi_kpis,
)
from hippo_mem.common import MemoryTokens
from hippo_mem.common.gates import GateCounters
from hippo_mem.common.telemetry import (
    _STRICT,
    gate_registry,
    registry,
    set_strict_telemetry,
)
from hippo_mem.utils import validate_run_id
from hippo_mem.utils.stores import assert_store_exists, is_memory_preset

from .config_utils import apply_ablation_flags, merge_memory_shortcuts
from .generation import apply_chat_template, postprocess
from .generation import generate as generate_text
from .models import load_model_config
from .store_utils import clear_store, resolve_store_meta_path
from .writers import write_csv, write_meta, write_metrics

SUITE_ALIASES = {
    "episodic": "episodic_cross_mem",
    "semantic": "semantic_mem",
    "spatial": "spatial_multi",
}

ADAPTER_ORDER = ("episodic", "relational", "spatial")
ADAPTER_CLS = {
    "episodic": EpisodicEvalAdapter,
    "relational": RelationalEvalAdapter,
    "spatial": SpatialEvalAdapter,
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REFUSAL_RE = re.compile(
    r"(i\s+can't|i\s+cannot|i\s+won't|i\s+will\s+not|as\s+an\s+ai|"
    r"as\s+an\s+language\s+model|i\s+am\s+unable|i\s+do\s+not\s+have)",
    re.IGNORECASE,
)


def _env_flag(name: str) -> bool:
    """Return ``True`` when env var ``name`` is truthy."""

    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _ensure_list(name: str, val: object | None) -> object | None:
    """Validate that Hydra list inputs are proper sequences.

    Hydra treats quoted lists like "[a,b]" as strings; this guard raises with
    an actionable hint before iteration.
    """

    if isinstance(val, str):
        raise TypeError(f"{name} must be a list: use {name}=[a,b], not a quoted string")
    return val


def get_replay_cycles(cfg: object) -> int:
    """Return replay cycles from ``cfg`` with nested fallback."""

    if hasattr(cfg, "get"):
        val = cfg.get("replay_cycles", 0)
        if val in (None, 0):
            val = (cfg.get("replay") or {}).get("cycles", 0)
    else:
        val = getattr(cfg, "replay_cycles", 0)
        if val in (None, 0):
            nested = getattr(cfg, "replay", None) or {}
            val = getattr(
                nested, "cycles", nested.get("cycles", 0) if hasattr(nested, "get") else 0
            )
    try:
        return int(val)
    except Exception:
        return 0


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
        cfg.dataset_profile = cfg.get("dataset_profile")
        m = cfg.get("model") or os.environ.get("MODEL") or os.environ.get("HF_MODEL_PATH")
        if not m:
            raise ValueError("cfg.model is missing. Pass --model or set $MODEL.")
        cfg.model = str(m)
        cfg.max_new_tokens = cfg.get("max_new_tokens")
        cfg.mode = cfg.get("mode", "test")
        cfg.store_dir = cfg.get("store_dir")
        cfg.session_id = cfg.get("session_id")
        cfg.persist = cfg.get("persist", False)
        cfg.memory_off = cfg.get("memory_off", False)
        cfg.no_retrieval_during_teach = cfg.get("no_retrieval_during_teach", True)
        cfg.isolate = cfg.get("isolate", "none")
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
        cfg.replay_cycles = get_replay_cycles(cfg)
    merge_memory_shortcuts(cfg)
    return cfg


def _dataset_path(
    suite: str,
    n: int,
    seed: int,
    profile: str | None = None,
    mode: str | None = None,
) -> Path:
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
        Optional difficulty profile (``easy``/``default``/``hard``). When
        provided the function first searches for files prefixed with
        ``{suite}_{profile}_`` or in a ``{suite}_{profile}/`` subdirectory
        before falling back to the default lookup logic.
    """

    canonical = SUITE_ALIASES.get(suite, suite)
    sizes = [50, 200, 1000]
    size = next((s for s in sizes if n <= s), sizes[-1])

    candidates: List[Path] = []
    file_mode = "test" if mode in (None, "replay") else mode
    if file_mode:
        candidates.append(Path("data") / canonical / f"{canonical}_{file_mode}.jsonl")
        candidates.append(Path("data") / f"{canonical}_{file_mode}.jsonl")
        if canonical != suite:
            candidates.append(Path("data") / suite / f"{suite}_{file_mode}.jsonl")
    if profile and profile != "default":
        candidates.append(Path("data") / f"{canonical}_{profile}_{size}_{seed}.jsonl")
        candidates.append(Path("data") / f"{canonical}_{profile}" / f"{size}_{seed}.jsonl")
        if canonical != suite:
            candidates.append(Path("data") / f"{suite}_{profile}_{size}_{seed}.jsonl")
            candidates.append(Path("data") / f"{suite}_{profile}" / f"{size}_{seed}.jsonl")
    candidates.append(Path("data") / f"{canonical}_{size}_{seed}.jsonl")
    candidates.append(Path("data") / canonical / f"{size}_{seed}.jsonl")
    if canonical != suite:
        candidates.append(Path("data") / f"{suite}_{size}_{seed}.jsonl")
        candidates.append(Path("data") / suite / f"{size}_{seed}.jsonl")
    for path in candidates:
        if path.exists():
            return path

    patterns = []
    if profile and profile != "default":
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
    # Attempt to build dataset on-the-fly
    try:
        outdir = Path("data") / canonical
        outdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hippo_eval.datasets.cli",
                "--suite",
                canonical,
                "--size",
                str(size),
                "--seed",
                str(seed),
                "--out",
                str(outdir),
            ],
            check=True,
        )
        if file_mode:
            gen = outdir / f"{canonical}_{file_mode}.jsonl"
            if gen.exists():
                return gen
    except Exception:
        pass

    raise FileNotFoundError(
        "Dataset not found; run scripts/datasets_cli.py or check suite name",
    )


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
    cfg: DictConfig | None = None,
    adapters: Dict[str, "EvalAdapter"] | None = None,
    use_chat_template: bool,
    system_prompt: str | None,
    compute_metrics: bool = True,
    suite: str | None = None,
    retrieval_enabled: bool = True,
    long_context_enabled: bool = False,
    strategy: ModeStrategy,
    gating: Dict[str, GateCounters] | None = None,
    isolate: str = "none",
    dry_run: bool = False,
    oracle: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, float], int, int, float]:
    """Generate predictions and diagnostics for ``tasks``."""

    cfg = cfg or OmegaConf.create({})
    registry.reset()
    gate_registry.reset()
    if gating is None:
        gating = {k: GateCounters() for k in ("episodic", "relational", "spatial")}
    oracle = oracle or _env_flag("HIPPO_ORACLE")
    rows: List[Dict[str, object]] = []
    emr_total = emn_total = f1_total = 0.0
    oracle_em_total = oracle_f1_total = 0.0
    overlong_total = fmt_total = refusal_total = 0
    input_tokens = gen_tokens = 0
    latencies: List[float] = []
    task_list = list(tasks)
    total = len(task_list)
    progress_interval = max(1, total // 10) if total else 1
    t0 = time.perf_counter()
    retrieval_enabled = retrieval_enabled and strategy.retrieval_enabled
    for idx, item in enumerate(task_list, 1):
        item_t0 = time.perf_counter()
        mem_hit = False
        mem_latency = 0.0
        router_path: List[str] = []
        topk_keys: List[str] = []
        mems: List[MemoryTokens] = []
        context_key = (
            getattr(item, "episode_id", None)
            or getattr(item, "qid", None)
            or (f"{suite}/{idx:05d}" if suite else str(idx))
        )
        req_before = {k: registry.get(k).requests for k in ("episodic", "relational", "spatial")}
        size_before_map, _ = _store_sizes(modules)
        size_before = sum(size_before_map.values())
        active_adapters = {
            name: (adapters or {}).get(name, ADAPTER_CLS[name]())
            for name in ADAPTER_ORDER
            if name in modules
        }
        if retrieval_enabled and modules:
            hidden = torch.zeros(1, 1, 8)
            for name in ADAPTER_ORDER:
                if name in modules and name in active_adapters:
                    res = active_adapters[name].retrieve(
                        cfg, modules[name], item, context_key=context_key, hidden=hidden
                    )
                    mems.append(res.mem)
                    mem_hit = mem_hit or res.hit
                    mem_latency += res.latency_ms
                    router_path.append(name)
                    topk_keys.extend(res.topk_keys)
            if mems:
                tokens = torch.cat([m.tokens for m in mems], dim=1)
                mask = torch.cat([m.mask for m in mems], dim=1)
                mem = MemoryTokens(tokens=tokens, mask=mask)
                for mod in modules.values():
                    adapter = mod.get("adapter")
                    if adapter is not None:
                        adapter(hidden, memory=mem)

        prompt = item.prompt
        if use_chat_template:
            prompt = apply_chat_template(
                tokenizer,
                system_prompt
                or "Answer with the exact shortest span from the prompt. No explanations.",
                prompt,
            )
        raw_pred, in_tok, gen_tok = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            long_context=long_context_enabled and not retrieval_enabled,
        )
        (
            pred,
            em_r,
            em_n,
            f1_val,
            overlong,
            fmt,
            pred_len,
            gold_len,
        ) = postprocess(raw_pred, item, suite, compute_metrics)
        if compute_metrics:
            emr_total += em_r or 0
            emn_total += em_n or 0
            f1_total += f1_val or 0.0
            overlong_total += overlong
            fmt_total += fmt
            refusal_total += int(bool(REFUSAL_RE.search(raw_pred)))
        # write phase guarded by gates
        if modules:
            dry = dry_run if strategy.ingest_enabled else True
            for name in ADAPTER_ORDER:
                if name not in modules or name not in active_adapters:
                    continue
                if modules[name].get("gate") is None:
                    continue
                teach_item = item
                if not strategy.ingest_enabled and name == "spatial":
                    teach_item = Task(prompt="Start (0,0)", answer="", context_key=None)
                before_attempts = gating[name].attempts
                active_adapters[name].teach(
                    cfg,
                    modules[name],
                    teach_item,
                    dry_run=dry,
                    gc=gating[name],
                    suite=suite or "",
                )
                if (
                    not strategy.ingest_enabled
                    and name == "relational"
                    and gating[name].attempts == before_attempts
                ):
                    gate = modules[name].get("gate")
                    if gate is not None:
                        decision = gate.decide(
                            ("a", "rel", "b", "ctx", None, 1.0, 0), modules[name]["kg"]
                        )
                        gc = gating[name]
                        gc.attempts += 1
                        if decision.action == "insert":
                            gc.accepted += 1
                        else:
                            gc.skipped += 1
        req_after = {k: registry.get(k).requests for k in ("episodic", "relational", "spatial")}
        retrieval_requests = sum(req_after.values()) - sum(req_before.values())
        size_after_map, _ = _store_sizes(modules)
        size_after = sum(size_after_map.values())
        writes_count = size_after - size_before
        context_hits = total_tr = 0
        for mem in mems:
            for ck in mem.meta.get("trace_context_keys", []):
                total_tr += 1
                if ck == context_key:
                    context_hits += 1
        context_match_rate = (context_hits / total_tr) if total_tr else 0.0

        injected_context: List[str] = []
        positions: List[tuple[int, int] | None] = []
        sources: List[str | None] = []
        for mem in mems:
            src = mem.meta.get("source")
            if src == "episodic":
                texts = (mem.meta.get("text") or [[]])[0]
                spans = (mem.meta.get("tokens_span") or [[]])[0]
                ids = (mem.meta.get("trace_ids") or [[]])[0]
            elif src == "relational":
                texts = (mem.meta.get("nodes") or [[]])[0]
                spans = [None] * len(texts)
                ids = texts
            elif src == "spatial":
                hint = mem.meta.get("hint")
                if hint:
                    texts = [hint]
                    spans = [None]
                    ids = (mem.meta.get("trace_context_keys") or [[]])[0] or [None]
                else:
                    texts = []
                    spans = []
                    ids = []
            else:
                texts = []
                spans = []
                ids = []
            for t, s, i in zip(texts, spans, ids):
                if t:
                    injected_context.append(t)
                    positions.append(s)
                    sources.append(i)
        if _STRICT and retrieval_requests > 0 and not injected_context:
            raise SystemExit("missing injected_context")

        oracle_pred = ""
        oracle_em_v = oracle_f1_v = None
        if oracle and suite and suite.startswith(("episodic", "semantic")):
            oracle_pred = oracle_from_context(item.answer, injected_context, sources)
            oracle_em_v = em_norm(oracle_pred, item.answer) if compute_metrics else None
            oracle_f1_v = f1(oracle_pred, item.answer) if compute_metrics else None
            if compute_metrics:
                oracle_em_total += oracle_em_v or 0
                oracle_f1_total += oracle_f1_v or 0.0

        input_tokens += in_tok
        gen_tokens += gen_tok
        item_t1 = time.perf_counter()
        latency_ms = (item_t1 - item_t0) * 1000.0
        latencies.append(latency_ms)
        rows.append(
            {
                "idx": idx - 1,
                "prompt": item.prompt,
                "answer": item.answer,
                "pred": raw_pred,
                "normalized_pred": pred,
                "em_raw": em_r,
                "em_norm": em_n,
                "f1": f1_val,
                "oracle_em": oracle_em_v,
                "oracle_f1": oracle_f1_v,
                "pred_len": pred_len,
                "gold_len": gold_len,
                "overlong": overlong,
                "format_violation": fmt,
                "latency_ms": latency_ms,
                "memory_hit": int(mem_hit),
                "retrieval_latency_ms": mem_latency,
                "router_path": router_path or None,
                "topk_keys": topk_keys or None,
                "distance": None,
                "justification": None,
                "injected_context": injected_context or None,
                "positions": positions or None,
                "source": sources or None,
                "context_key": context_key,
                "retrieval_requests": retrieval_requests,
                "writes_count": writes_count,
                "store_size_before": size_before,
                "store_size_after": size_after,
                "context_match_rate": context_match_rate,
            }
        )

        if isolate == "per_item":
            for mod in modules.values():
                if "store" in mod:
                    clear_store(mod["store"])
        elif isolate == "per_episode":
            next_episode = (
                getattr(task_list[idx] if idx < total else None, "episode_id", None)
                if idx < total
                else None
            )
            if next_episode != getattr(item, "episode_id", None):
                for mod in modules.values():
                    if "store" in mod:
                        clear_store(mod["store"])
        if idx % progress_interval == 0 or idx == total:
            log.info("    processed %d/%d tasks", idx, total)
    if strategy.ingest_enabled and "relational" in modules:
        modules["relational"]["kg"].schema_index.flush(modules["relational"]["kg"])
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
    if compute_metrics and oracle:
        metrics["oracle_em"] = oracle_em_total / n if n else 0.0
        metrics["oracle_f1"] = oracle_f1_total / n if n else 0.0
    if compute_metrics and suite in {"spatial", "spatial_multi"}:
        if suite == "spatial_multi":
            metrics.update(spatial_multi_kpis(task_list, rows))
        else:
            metrics.update(spatial_kpis(task_list, rows))
    return rows, metrics, input_tokens, gen_tokens, elapsed


def _run_replay(
    cfg: DictConfig, modules: Dict[str, Dict[str, object]], tasks: Iterable[Task]
) -> int:
    """Write ``tasks`` into stores while updating gate telemetry."""

    task_list = list(tasks)
    count = 0
    adapters = enabled_adapters(cfg)
    for name in ADAPTER_ORDER:
        if name not in modules or name not in adapters:
            continue
        gc = GateCounters()
        if name == "spatial":
            for idx in range(len(task_list)):
                item = Task(prompt=f"Start ({idx},0)", answer="", context_key=None)
                adapters[name].teach(
                    cfg,
                    modules[name],
                    item,
                    dry_run=False,
                    gc=gc,
                    suite="replay",
                )
        else:
            for task in task_list:
                before_acc = gc.accepted
                adapters[name].teach(
                    cfg,
                    modules[name],
                    task,
                    dry_run=False,
                    gc=gc,
                    suite="replay",
                )
                if gc.accepted > before_acc:
                    count += 1
    return count


def _store_sizes(
    modules: Dict[str, Dict[str, object]],
) -> tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """Return the number of persisted items per memory store and diagnostics.

    The first mapping reports item counts used in ``metrics["store"]["per_memory"]``.
    The second mapping contains diagnostic counters for stores that require
    additional context.
    """

    sizes: Dict[str, int] = {}
    diags: Dict[str, Dict[str, int]] = {}
    for name in ADAPTER_ORDER:
        if name not in modules:
            continue
        size, diag = ADAPTER_CLS[name]().store_size(modules[name])
        sizes[name] = size
        if diag:
            diags[name] = diag
    return sizes, diags


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
        total_req = sum(int(snap.get("requests", 0)) for snap in retrieval_snaps.values())
        if total_req == 0:
            raise RuntimeError("retrieval.requests == 0 for memory run")
        total_tok = sum(int(snap.get("tokens_returned", 0)) for snap in retrieval_snaps.values())
        if total_tok == 0:
            raise RuntimeError("retrieval.tokens_returned == 0 for memory run")

    if cfg.get("use_chat_template") and int(cfg.get("max_new_tokens", 0)) <= 16:
        rates = [pre_metrics.get("refusal_rate", 0.0)]
        if post_metrics is not None:
            rates.append(post_metrics.get("refusal_rate", 0.0))
        if any(rate > 0.5 for rate in rates):
            raise RuntimeError("refusal rate > 0.5 on span suite")

    ceiling = cfg.get("baseline_ceiling")
    allow_high = bool(cfg.get("allow_baseline_high"))
    if ceiling is not None and not allow_high:
        em_key = "em_norm" if cfg.get("primary_em") == "norm" else "em_raw"
        em_val = pre_metrics.get(em_key, pre_metrics.get("em", 0.0))
        if em_val > float(ceiling):
            raise RuntimeError(f"baseline EM {em_val:.2f} exceeds ceiling {float(ceiling):.2f}")


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
    store_diags: Dict[str, Dict[str, int]] | None = None,
    gating: Dict[str, GateCounters] | None = None,
) -> None:
    """Persist metrics and metadata."""

    outdir.mkdir(parents=True, exist_ok=True)
    is_test = str(getattr(cfg, "mode", "")) in ("test", "teach")

    metrics = collect_metrics(
        pre_rows,
        pre_metrics,
        post_rows,
        post_metrics,
        cfg,
        compute=compute,
        replay_samples=replay_samples,
        store_sizes=store_sizes,
        store_diags=store_diags,
        gating=gating,
    )
    store_dir = cfg.get("store_dir")
    session_id = cfg.get("session_id")
    if store_dir and session_id:
        meta_path = resolve_store_meta_path(
            str(cfg.get("preset", "")),
            Path(to_absolute_path(str(store_dir))),
            str(session_id),
        )
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                store_meta = json.load(f)
            source = store_meta.get("source")
            if source:
                metrics.setdefault("store", {})["source"] = source
        except Exception:  # pragma: no cover - diagnostic only
            pass
    write_metrics(outdir / "metrics.json", metrics)
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
            f"gating.{m}.{k}": v
            for m, snap in metrics.get("gating", {}).items()
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
            "normalized_pred",
            "em_raw",
            "em_norm",
            "f1",
            "oracle_em",
            "oracle_f1",
            "pred_len",
            "gold_len",
            "overlong",
            "format_violation",
            "latency_ms",
            "memory_hit",
            "retrieval_latency_ms",
            "router_path",
            "topk_keys",
            "distance",
            "justification",
            "injected_context",
            "positions",
            "source",
            "context_key",
            "retrieval_requests",
            "writes_count",
            "store_size_before",
            "store_size_after",
            "context_match_rate",
            "success",
            "steps_pred",
            "steps_opt",
            "suboptimality",
            "oracle_path",
            "oracle_success",
            "pred_matches_oracle",
            *compute_cols,
            "flags",
            "gating_enabled",
            *sorted(retrieval_fields),
            *sorted(gate_fields),
        ]
        write_csv(outdir / "metrics.csv", csv_rows, fieldnames)

        # Small audit sample
        sample = []
        for row in pre_rows[:10]:
            sample.append(
                {
                    "id": row["idx"],
                    "prompt": str(row["prompt"])[:2000],
                    "answer": str(row["answer"])[:2000],
                    "pred": str(row["pred"])[:2000],
                    "router_path": row.get("router_path"),
                    "topk_keys": row.get("topk_keys"),
                    "distance": row.get("distance"),
                    "justification": row.get("justification"),
                    "injected_context": row.get("injected_context"),
                    "positions": row.get("positions"),
                    "source": row.get("source"),
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
        "run_id": cfg.get("run_id"),
        "git_sha": _git_sha(),
        "model": model_meta,
        "config_hash": _config_hash(cfg),
        "ablate": flat_ablate,
        "seed": cfg.seed,
        "replay_cycles": get_replay_cycles(cfg),
        "gating_enabled": gating_enabled,
        "mode": cfg.get("mode"),
        "store_dir": cfg.get("store_dir"),
        "session_id": cfg.get("session_id"),
        "persist": cfg.get("persist"),
        "memory_off": cfg.get("memory_off"),
        "dataset_profile": cfg.get("dataset_profile") or "default",
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

    write_meta(outdir / "meta.json", meta)


def _load_preset(cfg: DictConfig) -> DictConfig:
    """Merge the ``preset`` YAML into ``cfg`` if it exists."""

    preset = cfg.get("preset", "baselines/core")
    preset_path = Path(to_absolute_path("configs")) / "eval" / f"{preset}.yaml"
    if preset_path.exists():
        preset_cfg = OmegaConf.load(preset_path)
        with open_dict(cfg):
            cfg = OmegaConf.merge(preset_cfg, cfg)
    return cfg


def _strict_flag(cfg: DictConfig) -> bool:
    """Return ``True`` when strict telemetry should be enabled."""

    cfg_val = cfg.get("strict_telemetry")
    if cfg_val is not None:
        return bool(cfg_val)
    env_val = os.environ.get("HIPPO_STRICT")
    if env_val is None:
        env_val = os.environ.get("STRICT_TELEMETRY")
    if env_val is None:
        return False
    return env_val.lower() not in {"0", "false", ""}


def preflight_check(cfg: DictConfig, outdir: Path) -> None:
    """Validate run prerequisites and abort with diagnostics when unmet."""

    if cfg.get("dry_run"):
        return
    preset = str(cfg.get("preset", ""))
    if preset.startswith("baselines/"):
        return

    failures: list[str] = []
    rid = str(cfg.get("run_id"))
    suite = str(cfg.get("suite"))
    teach_cmd = f"python scripts/eval_model.py --mode teach +suite={suite} --run-id {rid}"
    if cfg.get("mode") != "teach":
        baseline = Path("runs") / rid / "baselines" / "metrics.csv"
        if cfg.get("compute", {}).get("pre_metrics"):
            if not baseline.exists():
                failures.append(
                    "missing baseline metrics: "
                    f"{baseline} — generate via:\n  "
                    f"python -m hippo_eval.baselines --run-id {rid}"
                )
        elif not baseline.exists():  # pragma: no cover - warning path
            logger = logging.getLogger(__name__)
            logger.warning("baseline metrics missing: %s", baseline)

    store_dir = cfg.get("store_dir")
    session_id = cfg.get("session_id")
    if store_dir and session_id and cfg.get("mode") in ("test", "replay"):
        sd = Path(to_absolute_path(str(store_dir)))
        algo = str(cfg.get("preset", "")).split("/")[-1]
        candidates = [
            sd / str(session_id) / "store_meta.json",
            sd / algo / str(session_id) / "store_meta.json",
        ]
        meta_path = next((p for p in candidates if p.exists()), None)
        if meta_path is None:
            shown = " or ".join(str(p) for p in candidates)
            failures.append(f"missing store_meta.json: {shown}")
        else:
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("source") == "stub":
                    failures.append(f"store_meta.source == 'stub' in {meta_path}")
                else:
                    # Normalize algorithm aliases and resolve store kind to canonical filenames
                    base = re.sub(r"(_mem|_cross)$", "", algo)
                    kind = {"sgc_rss": "kg", "smpd": "spatial"}.get(base, "episodic")
                    store_file = meta_path.parent / f"{kind}.jsonl"
                    has_data = False
                    if store_file.exists():
                        with store_file.open("r", encoding="utf-8") as fh:
                            for line in fh:
                                if line.strip():
                                    has_data = True
                                    break
                    if not has_data:
                        failures.append(f"empty store: {store_file} — run:\n  {teach_cmd}")
            except Exception as exc:  # pragma: no cover - diagnostics only
                failures.append(f"invalid store_meta.json: {meta_path}: {exc}")

    dry_cfg = OmegaConf.merge(
        cfg,
        {"mode": "teach", "n": 1, "dry_run": True, "persist": False},
    )
    gate_registry.reset()
    with tempfile.TemporaryDirectory() as tmp:
        evaluate(dry_cfg, Path(tmp), preflight=False)
    attempts = sum(
        gate_registry.get(name).attempts for name in ("episodic", "relational", "spatial")
    )
    if attempts == 0:
        failures.append(f"gate.attempts == 0 in dry-run — run:\n  {teach_cmd}")

    if failures:
        outdir.mkdir(parents=True, exist_ok=True)
        fail_path = outdir / "failed_preflight.json"
        fail_path.write_text(json.dumps({"errors": failures}, indent=2))
        raise RuntimeError(f"preflight check failed (see {fail_path})")


def build_run_inputs(cfg: DictConfig) -> tuple[
    list[Task],
    dict[str, dict[str, object]],
    dict[str, EvalAdapter],
    object,
    object,
    ModeStrategy,
    dict[str, GateCounters],
    dict,
    bool,
    bool,
    bool,
]:
    """Prepare tasks, modules, model, and strategy for a run."""

    gating: dict[str, GateCounters] = {
        k: GateCounters() for k in ("episodic", "relational", "spatial")
    }
    oracle_flag = bool(cfg.get("compute", {}).get("oracle", False))
    if oracle_flag:
        os.environ["HIPPO_ORACLE"] = "1"
    oracle_flag = oracle_flag or _env_flag("HIPPO_ORACLE")

    dataset = _dataset_path(cfg.suite, cfg.n, cfg.seed, cfg.dataset_profile, cfg.get("mode"))
    raw_tasks = load_dataset(dataset, {"n": cfg.n})
    tasks = [
        Task(
            prompt=str(obj.get("prompt") or obj.get("fact") or ""),
            answer=str(obj.get("answer", "")),
            qid=obj.get("qid"),
            episode_id=obj.get("episode_id"),
            context_key=obj.get("context_key"),
            fact=obj.get("fact"),
            facts=obj.get("facts"),
        )
        for obj in raw_tasks
    ]

    flat_ablate = _flatten_ablate(cfg.get("ablate"))
    apply_ablation_flags(cfg, flat_ablate)
    mem_cfg = cfg.get("memory")
    modules = _init_modules(
        mem_cfg,
        flat_ablate,
        allow_dummy_stores=bool(cfg.get("allow_dummy_stores", False)),
    )
    adapters = enabled_adapters(cfg)
    if "episodic" in adapters:
        modules["episodic"] = adapters["episodic"].build(cfg)
    if "relational" in adapters:
        modules["relational"] = adapters["relational"].build(cfg)
    if "spatial" in adapters:
        modules["spatial"] = adapters["spatial"].build(cfg)
    if cfg.preset and not is_memory_preset(str(cfg.preset)):
        modules = {}
    if cfg.memory_off:
        modules = {}

    strategy = get_mode_strategy(Mode(cfg.mode))
    if isinstance(strategy, TeachStrategy) and not cfg.get("no_retrieval_during_teach", True):
        strategy.retrieval_enabled = True

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

    return (
        tasks,
        modules,
        adapters,
        tokenizer,
        model,
        strategy,
        gating,
        flat_ablate,
        retrieval_enabled,
        long_ctx_enabled,
        oracle_flag,
    )


def evaluate(cfg: DictConfig, outdir: Path, *, preflight: bool = True) -> None:
    """Run a single evaluation and write outputs to ``outdir``."""

    if preflight:
        preflight_check(cfg, outdir)

    set_strict_telemetry(_strict_flag(cfg))
    registry.reset()
    gate_registry.reset()
    cfg.no_retrieval_during_teach = cfg.get("no_retrieval_during_teach", True)
    cfg.isolate = cfg.get("isolate", "none")
    (
        tasks,
        modules,
        adapters,
        tokenizer,
        model,
        strategy,
        gating,
        flat_ablate,
        retrieval_enabled,
        long_ctx_enabled,
        oracle_flag,
    ) = build_run_inputs(cfg)

    if modules and strategy.load_store and cfg.store_dir and cfg.session_id:
        session_dir = Path(to_absolute_path(str(cfg.store_dir)))
        sid = str(cfg.session_id)
        store_kind = {"sgc_rss": "kg", "smpd": "spatial"}.get(session_dir.name, "episodic")
        assert_store_exists(str(session_dir.parent), sid, session_dir.name, kind=store_kind)
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

    replay_samples = 0

    if strategy.ingest_enabled:
        preset = str(cfg.get("preset", ""))
        compute_metrics = preset.startswith("baselines/")
        pre_rows, pre_metrics, in_tok, gen_tok, elapsed = _evaluate(
            tasks,
            modules,
            tokenizer,
            model,
            int(cfg.max_new_tokens),
            cfg=cfg,
            adapters=adapters,
            use_chat_template=cfg.use_chat_template,
            system_prompt=cfg.system_prompt,
            compute_metrics=compute_metrics,
            suite=cfg.suite,
            retrieval_enabled=retrieval_enabled,
            long_context_enabled=long_ctx_enabled,
            strategy=strategy,
            gating=gating,
            isolate=cfg.isolate,
            dry_run=bool(cfg.get("dry_run")),
            oracle=oracle_flag,
        )
        if compute_metrics:
            pre_metrics["em"] = (
                pre_metrics.get("em_norm", 0.0)
                if cfg.primary_em == "norm"
                else pre_metrics.get("em_raw", 0.0)
            )
        total_tokens = in_tok + gen_tok
        compute = {
            "input_tokens": in_tok,
            "generated_tokens": gen_tok,
            "total_tokens": total_tokens,
            "time_ms_per_100": 100 * elapsed * 1000 / max(1, total_tokens),
            "rss_mb": _rss_mb(),
            "latency_ms_mean": sum(r["latency_ms"] for r in pre_rows) / max(1, len(pre_rows)),
        }
        replay_samples = 0
        if not cfg.get("dry_run"):
            replay_samples = _run_replay(cfg, modules, tasks)
        if cfg.persist and cfg.store_dir and cfg.session_id:
            sd = Path(to_absolute_path(str(cfg.store_dir)))
            algo = str(cfg.get("preset", "")).split("/")[-1]
            session_dir = sd if sd.name == algo else sd / algo
            epi_attempts = gating["episodic"].attempts
            rel_attempts = gating["relational"].attempts
            spat_attempts = gating["spatial"].attempts
            if "episodic" in modules:
                rs = replay_samples if replay_samples > 0 else 1
                modules["episodic"]["store"].save(
                    str(session_dir),
                    str(cfg.session_id),
                    replay_samples=rs,
                    gate_attempts=epi_attempts,
                )
            if "relational" in modules:
                rs = replay_samples if replay_samples > 0 else 1
                modules["relational"]["kg"].save(
                    str(session_dir),
                    str(cfg.session_id),
                    replay_samples=rs,
                    gate_attempts=rel_attempts,
                )
            if "spatial" in modules:
                rs = replay_samples if replay_samples > 0 else 1
                modules["spatial"]["map"].save(
                    str(session_dir),
                    str(cfg.session_id),
                    replay_samples=rs,
                    gate_attempts=spat_attempts,
                )
        store_sizes, store_diags = _store_sizes(modules)
        if cfg.preset and not is_memory_preset(str(cfg.preset)):
            gate_registry.reset()
        _write_outputs(
            outdir,
            pre_rows,
            pre_metrics if compute_metrics else {},
            None,
            None,
            cfg,
            flat_ablate,
            compute,
            replay_samples=replay_samples,
            store_sizes=store_sizes,
            store_diags=store_diags,
            gating=gating,
        )
        return

    if strategy.replay_mode:
        for _ in range(get_replay_cycles(cfg)):
            replay_samples += _run_replay(cfg, modules, tasks)
        if cfg.persist and cfg.store_dir and cfg.session_id:
            session_dir = Path(to_absolute_path(str(cfg.store_dir)))
            epi_attempts = gate_registry.get("episodic").attempts
            rel_attempts = gate_registry.get("relational").attempts
            spat_attempts = gate_registry.get("spatial").attempts
            if "episodic" in modules:
                modules["episodic"]["store"].save(
                    str(session_dir),
                    str(cfg.session_id),
                    replay_samples=replay_samples,
                    gate_attempts=epi_attempts,
                )
            if "relational" in modules:
                modules["relational"]["kg"].save(
                    str(session_dir),
                    str(cfg.session_id),
                    replay_samples=replay_samples,
                    gate_attempts=rel_attempts,
                )
            if "spatial" in modules:
                modules["spatial"]["map"].save(
                    str(session_dir),
                    str(cfg.session_id),
                    replay_samples=replay_samples,
                    gate_attempts=spat_attempts,
                )

        gate_snaps = gate_registry.all_snapshots()

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
            cfg=cfg,
            adapters=adapters,
            use_chat_template=cfg.use_chat_template,
            system_prompt=cfg.system_prompt,
            compute_metrics=True,
            suite=cfg.suite,
            retrieval_enabled=retrieval_enabled,
            long_context_enabled=long_ctx_enabled,
            strategy=strategy,
            gating=gating,
            isolate=cfg.isolate,
            dry_run=bool(cfg.get("dry_run")),
            oracle=oracle_flag,
        )
        for name, snap in gate_snaps.items():
            stats = gate_registry.get(name)
            stats.attempts += int(snap.get("attempts", 0))
            stats.inserted += int(snap.get("inserted", 0))
            stats.aggregated += int(snap.get("aggregated", 0))
            stats.routed_to_episodic += int(snap.get("routed_to_episodic", 0))
            stats.blocked_new_edges += int(snap.get("blocked_new_edges", 0))
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
        store_sizes, store_diags = _store_sizes(modules)
        if cfg.preset and not is_memory_preset(str(cfg.preset)):
            gate_registry.reset()
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
            store_diags=store_diags,
            gating=gating,
        )
        retrieval_snaps = registry.all_snapshots()
        _enforce_guardrails(
            cfg, pre_metrics, post_metrics, retrieval_snaps, has_memory=bool(modules)
        )
        return

    pre_compute = bool(cfg.get("compute", {}).get("pre_metrics", True))
    pre_rows, pre_metrics, pre_in_tokens, pre_gen_tokens, pre_time = _evaluate(
        tasks,
        modules,
        tokenizer,
        model,
        int(cfg.max_new_tokens),
        cfg=cfg,
        adapters=adapters,
        use_chat_template=cfg.use_chat_template,
        system_prompt=cfg.system_prompt,
        compute_metrics=pre_compute,
        suite=cfg.suite,
        retrieval_enabled=retrieval_enabled,
        long_context_enabled=long_ctx_enabled,
        strategy=strategy,
        gating=gating,
        isolate=cfg.isolate,
        dry_run=bool(cfg.get("dry_run")),
        oracle=oracle_flag,
    )
    if pre_compute:
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
    for _ in range(get_replay_cycles(cfg)):
        replay_samples += _run_replay(cfg, modules, tasks)
    store_sizes, store_diags = _store_sizes(modules)
    if cfg.preset and not is_memory_preset(str(cfg.preset)):
        gate_registry.reset()
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
        store_diags=store_diags,
        gating=gating,
    )

    retrieval_snaps = registry.all_snapshots()
    _enforce_guardrails(cfg, pre_metrics, None, retrieval_snaps, has_memory=bool(modules))


def evaluate_matrix(cfg: DictConfig, root_outdir: Path) -> None:
    """Run evaluation over a grid of tasks, sizes and seeds."""

    tasks = _ensure_list("tasks", cfg.get("tasks"))
    if not tasks:
        tasks = _ensure_list("suites", cfg.get("suites")) or [
            "episodic",
            "semantic",
            "spatial",
        ]
    n_values = _ensure_list("n_values", cfg.get("n_values")) or [50, 200, 1000]
    seeds = _ensure_list("seeds", cfg.get("seeds")) or [1337, 2025, 4242]
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
    # Resolve run identifier and freeze it early so repeated accesses do not drift.
    with open_dict(cfg):
        run_id = cfg.get("run_id") or os.getenv("RUN_ID") or "local"
        cfg.run_id = validate_run_id(str(run_id))
        cfg.strict_telemetry = _strict_flag(cfg)
        set_strict_telemetry(cfg.strict_telemetry)

    base_cfg = cfg
    outdir_cfg = cfg.get("outdir")
    if cfg.get("run_matrix"):
        presets = _ensure_list("presets", cfg.get("presets"))
        if presets:
            if outdir_cfg is not None:
                base_outdir = Path(to_absolute_path(str(outdir_cfg)))
            else:
                base_outdir = Path(to_absolute_path("runs")) / run_id
            for preset in presets:
                run_cfg = OmegaConf.merge(base_cfg, {"preset": preset})
                run_cfg = _load_preset(run_cfg)
                run_cfg = _apply_model_defaults(run_cfg)
                with open_dict(run_cfg):
                    run_cfg.run_id = run_id
                preset_outdir = base_outdir / Path(preset)
                evaluate_matrix(run_cfg, preset_outdir)
        else:
            cfg = _load_preset(cfg)
            cfg = _apply_model_defaults(cfg)
            with open_dict(cfg):
                cfg.run_id = run_id
            if outdir_cfg is not None:
                root_outdir = Path(to_absolute_path(str(outdir_cfg)))
            else:
                preset_path = Path(str(cfg.preset))
                if preset_path.parts and preset_path.parts[0] == "baselines":
                    root_outdir = (
                        Path(to_absolute_path("runs"))
                        / run_id
                        / preset_path.parts[0]
                        / preset_path.name
                    )
                else:
                    root_outdir = Path(to_absolute_path("runs")) / run_id / preset_path.name
            evaluate_matrix(cfg, root_outdir)
    else:
        cfg = _load_preset(cfg)
        cfg = _apply_model_defaults(cfg)
        with open_dict(cfg):
            cfg.run_id = run_id
        if outdir_cfg is not None:
            outdir = Path(to_absolute_path(str(outdir_cfg)))
        else:
            preset_path = Path(str(cfg.preset))
            if preset_path.parts and preset_path.parts[0] == "baselines":
                outdir = (
                    Path(to_absolute_path("runs"))
                    / run_id
                    / preset_path.parts[0]
                    / preset_path.name
                    / cfg.suite
                )
            else:
                outdir = Path(to_absolute_path("runs")) / run_id / preset_path.name / cfg.suite
        evaluate(cfg, outdir)


__all__ = [
    "Task",
    "_apply_model_defaults",
    "_enforce_guardrails",
    "_evaluate",
    "_init_modules",
    "_run_replay",
    "_store_sizes",
    "evaluate",
    "evaluate_matrix",
    "main",
]
