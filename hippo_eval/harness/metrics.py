"""Metric aggregation helpers for the evaluation harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from hippo_mem.common.gates import GateCounters
from hippo_mem.common.telemetry import registry


@dataclass
class MetricSchema:
    """Dataclass wrapper around a metrics mapping."""

    data: Dict[str, object]


def collect_metrics(
    pre_rows: List[Dict[str, object]],
    pre_metrics: Dict[str, float],
    post_rows: Optional[List[Dict[str, object]]],
    post_metrics: Optional[Dict[str, float]],
    cfg,
    *,
    compute: Dict[str, float] | None = None,
    replay_samples: int = 0,
    store_sizes: Dict[str, int] | None = None,
    store_diags: Dict[str, Dict[str, int]] | None = None,
    gating: Dict[str, GateCounters] | None = None,
) -> Dict[str, object]:
    """Return a metrics dictionary following ``metrics.json`` schema."""

    suite_metrics: Dict[str, float | None] = {
        "pre_em": pre_metrics.get("em"),
        "pre_em_raw": pre_metrics.get("em_raw"),
        "pre_em_norm": pre_metrics.get("em_norm"),
        "pre_f1": pre_metrics.get("f1"),
        "pre_refusal_rate": pre_metrics.get("refusal_rate"),
        "memory_hit_rate": 0.0,
        "latency_ms_delta": 0.0,
    }
    for k in ("success_rate", "suboptimality_ratio", "steps_to_goal"):
        if k in pre_metrics:
            suite_metrics[f"pre_{k}"] = pre_metrics[k]
    diagnostics: Dict[str, int | None] = {
        "pre_overlong": pre_metrics.get("overlong"),
        "pre_format_violation": pre_metrics.get("format_violation"),
    }
    if pre_rows:
        hits = sum(int(r.get("memory_hit", 0)) for r in pre_rows)
        lat_delta = sum(float(r.get("retrieval_latency_ms", 0.0)) for r in pre_rows)
        suite_metrics["memory_hit_rate"] = hits / max(1, len(pre_rows))
        suite_metrics["latency_ms_delta"] = lat_delta / max(1, len(pre_rows))
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
        "version": 2,
        "phase": str(getattr(cfg, "mode", "test")),
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "dataset_profile": cfg.get("dataset_profile") or "default",
        "metrics": {cfg.suite: suite_metrics},
        "diagnostics": {cfg.suite: diagnostics},
        "retrieval": registry.all_snapshots(),
        "gating": {k: asdict(v) for k, v in (gating or {}).items()},
        "replay": {"samples": replay_samples},
        "store": {
            "size": sum((store_sizes or {}).values()),
            "per_memory": store_sizes or {},
            "diagnostics": store_diags or {},
        },
    }
    if compute:
        metrics["metrics"]["compute"] = compute
    return metrics
