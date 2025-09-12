# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import os
import sys
from typing import Dict, List

from omegaconf import DictConfig


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


def summarize(
    cfg: DictConfig,
    metrics_pre: Dict[str, object],
    modules: Dict[str, Dict[str, object]],
    total_in_tokens: int,
    total_gen_tokens: int,
    total_time: float,
    lat_sum: float,
    total_items: int,
    post_metrics: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, object]:
    """Combine pre metrics and compute totals for a run."""

    mem_usage = _memory_usage(modules)
    total_tokens = total_in_tokens + total_gen_tokens
    avg_latency = lat_sum / max(1, total_items)
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
    return metrics


def aggregate_across_runs(results: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate summaries from multiple runs."""

    return {"runs": len(results)}


__all__ = ["summarize", "aggregate_across_runs"]
