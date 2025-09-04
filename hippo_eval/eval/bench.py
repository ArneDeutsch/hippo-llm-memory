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
to ``runs/<run_id>/<preset>/<suite>/`` for baseline presets such as
``baselines/core``.  For memory presets (``memory/hei_nw`` etc.) the files are
stored under ``runs/<run_id>/<preset_name>/<suite>/`` where ``preset_name`` is the
final path component.  A custom root directory can be supplied via
``outdir=...``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict

from hippo_eval.eval.score import em_norm, em_raw, f1
from hippo_mem.adapters import (
    EpisodicMemoryAdapter,
    RelationalMemoryAdapter,
    SpatialMemoryAdapter,
)
from hippo_mem.common.telemetry import gate_registry, registry
from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic.adapter import AdapterConfig
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.adapter import AdapterConfig as SpatialAdapterConfig
from hippo_mem.spatial.place_graph import PlaceGraph

from .datasets import SUITE_TO_GENERATOR


def _date_str(value: object | None) -> str:
    """Return a normalized date string consistent with ``eval.harness``."""

    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    date = str(value)
    if "_" not in date and date.isdigit() and len(date) > 8:
        return f"{date[:8]}_{date[8:]}"
    return date


FORMAT_VIOL_RE = re.compile(r"\n|\.$")

SLUG_RE = re.compile(r"^[A-Za-z0-9._-]{3,64}$")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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


def _pip_hash() -> str:
    """Return a SHA256 of ``pip freeze`` output."""

    try:
        out = subprocess.check_output(["pip", "freeze"], text=True)
    except Exception:  # pragma: no cover - pip unavailable
        return "unknown"
    return hashlib.sha256(out.encode("utf-8")).hexdigest()


def _cpu_info() -> str:
    """Return the CPU model string if available."""

    info = platform.processor()
    if not info:
        info = platform.machine()
    return info or "unknown"


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


def _sha256_file(path: Path) -> str:
    """Compute SHA256 for ``path``."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_tasks(suite: str, n: int, seed: int) -> List[Dict[str, object]]:
    """Load tasks from disk if available, otherwise generate."""

    data_path = Path("data") / suite / f"{n}_{seed}.jsonl"
    if data_path.exists():
        checksum_path = data_path.parent / "checksums.json"
        if not checksum_path.exists():
            raise FileNotFoundError(f"Missing {checksum_path}")
        with checksum_path.open("r", encoding="utf-8") as f:
            checksums = json.load(f)
        digest = _sha256_file(data_path)
        if checksums.get(data_path.name) != digest:
            raise RuntimeError(f"Checksum mismatch for {data_path}")
        with data_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return generate_tasks(suite, n, seed)


def _init_modules(
    memory: Optional[object],
    ablate: Dict[str, object],
    *,
    allow_dummy_stores: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Initialise memory modules based on ``memory`` config.

    ``memory`` may be a single name, list of names or a dict with keys
    ``episodic``, ``relational`` and ``spatial``. ``ablate`` contains flattened
    ablation flags which can disable optional components such as Hopfield or
    product quantisation. When ``allow_dummy_stores`` is ``False`` no placeholder
    records are written during initialisation.
    """

    modules: Dict[str, Dict[str, object]] = {}
    if not memory:
        return modules

    def _add_episodic() -> None:
        hopfield = bool(ablate.get("memory.episodic.hopfield", True))
        pq = bool(ablate.get("memory.episodic.pq", True))
        store = EpisodicStore(dim=8, config={"hopfield": hopfield, "pq": pq})
        if allow_dummy_stores:
            store.write(np.ones(8, dtype="float32"), "dummy")
        adapter_cfg = AdapterConfig(hidden_size=8, num_heads=1, enabled=True)
        modules["episodic"] = {"store": store, "adapter": EpisodicMemoryAdapter(adapter_cfg)}

    def _add_relational() -> None:
        kg = KnowledgeGraph()
        if allow_dummy_stores:
            kg.upsert("a", "rel", "b", "a rel b")
        modules["relational"] = {"kg": kg, "adapter": RelationalMemoryAdapter()}

    def _add_spatial() -> None:
        g = PlaceGraph()
        if allow_dummy_stores:
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
    retrieval_enabled: bool,
    long_context_enabled: bool,
) -> tuple[List[Dict[str, object]], Dict[str, float], int, float, float, float]:
    """Evaluate ``tasks`` returning rows, metrics and token stats."""

    rows: List[Dict[str, object]] = []
    emr_total = emn_total = f1_total = 0.0
    overlong_total = fmt_total = 0
    input_tokens = gen_tokens = 0
    t0 = time.perf_counter()
    latencies: List[float] = []
    for off, item in enumerate(tasks):
        item_t0 = time.perf_counter()
        epi_feats = None
        if retrieval_enabled and "episodic" in modules:
            store = modules["episodic"]["store"]
            traces = store.recall(np.zeros(8, dtype="float32"), 1)
            if traces:
                mem = torch.tensor([t.key for t in traces], dtype=torch.float32).unsqueeze(0)
            else:
                mem = torch.zeros(1, 1, 8)
            hidden_t = torch.zeros(1, 1, 8)
            epi_feats = modules["episodic"]["adapter"](hidden_t, mem).detach().numpy()[0]
        if retrieval_enabled and "relational" in modules:
            modules["relational"]["kg"].retrieve(np.zeros(8, dtype="float32"))
            modules["relational"]["adapter"](
                np.zeros(8, dtype="float32"),
                np.zeros((1, 8), dtype="float32"),
                epi_feats,
            )
        if retrieval_enabled and "spatial" in modules:
            modules["spatial"]["map"].plan("a", "b")
            plans = torch.zeros(1, 1, 8)
            modules["spatial"]["adapter"](torch.zeros(1, 1, 8), plans)

        prompt = str(item["prompt"])
        if long_context_enabled and not retrieval_enabled:
            prompt = f"{prompt} [CTX]"
        answer = str(item["answer"])
        pred = str(item.get("pred", answer))

        pred_len = len(pred.split())
        gold_len = len(answer.split())
        overlong = int(pred_len > gold_len)
        fmt = int(bool(FORMAT_VIOL_RE.search(pred)))
        em_r = em_raw(pred, answer)
        em_n = em_norm(pred, answer)
        f1_val = f1(pred, answer)
        emr_total += em_r
        emn_total += em_n
        f1_total += f1_val
        overlong_total += overlong
        fmt_total += fmt
        input_tokens += len(prompt.split())
        gen_tokens += len(answer.split())
        item_t1 = time.perf_counter()
        latency_ms = (item_t1 - item_t0) * 1000.0
        latencies.append(latency_ms)
        rows.append(
            {
                "idx": start_idx + off,
                "prompt": prompt,
                "answer": answer,
                "pred": pred,
                "em_raw": em_r,
                "em_norm": em_n,
                "f1": f1_val,
                "pred_len": pred_len,
                "gold_len": gold_len,
                "overlong": overlong,
                "format_violation": fmt,
                "latency_ms": latency_ms,
                "memory_hit": 0,
                "retrieval_latency_ms": 0.0,
                "flags": flag,
            }
        )
    t1 = time.perf_counter()
    if all(lat == 0.0 for lat in latencies) and latencies:
        fallback = (t1 - t0) * 1000.0 / len(latencies)
        for row in rows:
            row["latency_ms"] = fallback
        latencies = [fallback] * len(rows)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    n = len(tasks)
    metrics = {
        "em_raw": emr_total / n if n else 0.0,
        "em_norm": emn_total / n if n else 0.0,
        "f1": f1_total / n if n else 0.0,
        "overlong": overlong_total,
        "format_violation": fmt_total,
    }
    return rows, metrics, input_tokens, gen_tokens, t1 - t0, avg_latency


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

    tasks = _load_tasks(cfg.suite, cfg.n, cfg.seed)
    flat_ablate = _flatten_ablate(cfg.get("ablate"))
    modules = _init_modules(
        cfg.get("memory"),
        flat_ablate,
        allow_dummy_stores=bool(cfg.get("allow_dummy_stores", False)),
    )

    retrieval_enabled = bool(cfg.get("retrieval", {}).get("enabled", False))
    long_ctx_enabled = bool(cfg.get("long_context", {}).get("enabled", False))

    registry.reset()
    gate_registry.reset()
    rows, metrics_pre, in_tokens, gen_tokens, elapsed, lat_mean = _eval_tasks(
        tasks,
        modules,
        flag="pre_replay",
        start_idx=0,
        retrieval_enabled=retrieval_enabled,
        long_context_enabled=long_ctx_enabled,
    )
    hits = sum(int(r.get("memory_hit", 0)) for r in rows)
    lat_delta = sum(float(r.get("retrieval_latency_ms", 0.0)) for r in rows)
    metrics_pre.setdefault("memory_hit_rate", hits / max(1, len(rows)))
    metrics_pre.setdefault("latency_ms_delta", lat_delta / max(1, len(rows)))
    total_time = elapsed
    lat_sum = lat_mean * len(tasks)
    total_in_tokens = in_tokens
    total_gen_tokens = gen_tokens

    post_cycles = int(cfg.get("post_replay_cycles", 0))
    post_metrics: Dict[str, Dict[str, object]] = {}
    if post_cycles > 0:
        start = len(tasks)
        for cycle in range(1, post_cycles + 1):
            _run_replay_once(modules)
            registry.reset()
            gate_registry.reset()
            (
                cycle_rows,
                cycle_metrics,
                cycle_in_tokens,
                cycle_gen_tokens,
                cycle_time,
                cycle_lat,
            ) = _eval_tasks(
                tasks,
                modules,
                flag=f"post_replay_{cycle}",
                start_idx=start,
                retrieval_enabled=retrieval_enabled,
                long_context_enabled=long_ctx_enabled,
            )
            rows.extend(cycle_rows)
            post_tokens = cycle_in_tokens + cycle_gen_tokens
            post_metrics[str(cycle)] = {
                cfg.suite: cycle_metrics,
                "compute": {
                    "input_tokens": cycle_in_tokens,
                    "generated_tokens": cycle_gen_tokens,
                    "total_tokens": post_tokens,
                    "time_ms_per_100": 100 * cycle_time * 1000 / max(1, post_tokens),
                    "rss_mb": _rss_mb(),
                    "latency_ms_mean": cycle_lat,
                },
            }
            total_in_tokens += cycle_in_tokens
            total_gen_tokens += cycle_gen_tokens
            total_time += cycle_time
            lat_sum += cycle_lat * len(tasks)
            start += len(tasks)

    mem_usage = _memory_usage(modules)
    total_items = len(rows)
    avg_latency = lat_sum / max(1, total_items)
    total_tokens = total_in_tokens + total_gen_tokens
    metrics = {
        "version": 2,
        "phase": str(getattr(cfg, "mode", "test")),
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "metrics": {
            cfg.suite: metrics_pre,
            "compute": {
                "input_tokens": total_in_tokens,
                "generated_tokens": total_gen_tokens,
                "total_tokens": total_tokens,
                "time_ms_per_100": 100 * total_time * 1000 / max(1, total_tokens),
                "rss_mb": _rss_mb(),
                "latency_ms_mean": avg_latency,
            },
        },
        "memory": mem_usage,
        "retrieval": {},
        "gating": {},
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
        compute = metrics.get("metrics", {}).get("compute", {})
        compute_cols = [k for k in ("time_ms_per_100", "rss_mb") if k in compute]
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
            "memory_hit",
            "retrieval_latency_ms",
            *compute_cols,
            "flags",
            "gating_enabled",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            for col in compute_cols:
                row[col] = compute.get(col)
            row["gating_enabled"] = gating_enabled
            writer.writerow(row)

    cfg_hash = _config_hash(cfg)
    meta = {
        "suite": cfg.suite,
        "preset": cfg.preset,
        "n": cfg.n,
        "git_sha": _git_sha(),
        "model": cfg.get("model", "mock"),
        "config_hash": cfg_hash,
        "ablate": flat_ablate,
        "seed": cfg.seed,
        "gating_enabled": gating_enabled,
        "python": sys.version,
        "platform": platform.platform(),
        "pip_hash": _pip_hash(),
        "cpu": _cpu_info(),
    }
    if torch.cuda.is_available():
        cuda_meta = {"version": torch.version.cuda}
        if hasattr(torch._C, "_cuda_getDriverVersion"):
            cuda_meta["driver"] = torch._C._cuda_getDriverVersion()
        meta["cuda"] = cuda_meta
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
    registry.reset()
    gate_registry.reset()
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


def main(cfg: DictConfig) -> None:
    """Run the evaluation bench with ``cfg``."""

    cfg.preset = cfg.get("preset", "baselines/core")
    cfg.seed = cfg.get("seed", 0)
    cfg.n = cfg.get("n", 5)
    if cfg.get("dry_run"):
        cfg.n = min(cfg.n, 5)
    run_id = cfg.get("run_id")
    if not run_id and cfg.get("date"):
        run_id = _date_str(cfg.get("date"))
        log.warning("`date` is deprecated for IO; using run_id=%s", run_id)
    if not run_id:
        run_id = _date_str(None)
    run_id = str(run_id)
    if not SLUG_RE.match(run_id):
        raise ValueError("run_id must match ^[A-Za-z0-9._-]{3,64}$")
    with open_dict(cfg):
        cfg.run_id = run_id
        cfg.date = cfg.get("date")
    outdir: Optional[str] = cfg.get("outdir")
    preset_path = Path(str(cfg.preset))
    if cfg.get("run_matrix"):
        if outdir is not None:
            root_outdir = Path(to_absolute_path(outdir))
        else:
            if preset_path.parts and preset_path.parts[0] == "baselines":
                root_outdir = Path("runs") / run_id / preset_path.parts[0] / preset_path.name
            else:
                root_outdir = Path("runs") / run_id / preset_path.name
        evaluate_matrix(cfg, root_outdir)
    else:
        if outdir is not None:
            outdir_path = Path(to_absolute_path(outdir))
        else:
            if preset_path.parts and preset_path.parts[0] == "baselines":
                outdir_path = (
                    Path("runs") / run_id / preset_path.parts[0] / preset_path.name / cfg.suite
                )
            else:
                outdir_path = Path("runs") / run_id / preset_path.name / cfg.suite
        evaluate(cfg, outdir_path)


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="default")
def cli(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point forwarding to :func:`main`."""

    main(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    cli()
