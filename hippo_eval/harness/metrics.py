"""Metric aggregation helpers for the evaluation harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from hippo_mem.common.gates import GateCounters
from hippo_mem.common.telemetry import registry


@dataclass
class MetricSchema:
    """Dataclass wrapper around a metrics mapping."""

    data: Dict[str, object]


@dataclass
class SuiteAccumulator:
    """Mutable container for suite metrics and diagnostics."""

    metrics: Dict[str, float | None] = field(default_factory=dict)
    diagnostics: Dict[str, int | None] = field(default_factory=dict)

    def update_post(self, post_metrics: Dict[str, float]) -> None:
        """Merge post-suite metrics and diagnostics."""

        self.metrics.update(
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
                self.metrics[f"post_{k}"] = post_metrics[k]
        self.diagnostics.update(
            {
                "post_overlong": post_metrics.get("overlong", 0),
                "post_format_violation": post_metrics.get("format_violation", 0),
            }
        )


def init_suite_metrics(pre_metrics: Dict[str, float]) -> SuiteAccumulator:
    """Initialize accumulator with pre-suite metrics and diagnostics."""

    metrics: Dict[str, float | None] = {
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
            metrics[f"pre_{k}"] = pre_metrics[k]
    diagnostics = {
        "pre_overlong": pre_metrics.get("overlong"),
        "pre_format_violation": pre_metrics.get("format_violation"),
    }
    return SuiteAccumulator(metrics=metrics, diagnostics=diagnostics)


def compute_deltas(
    acc: SuiteAccumulator, pre_metrics: Dict[str, float], post_metrics: Dict[str, float]
) -> None:
    """Populate ``delta_*`` metrics comparing pre and post suites."""

    for key, pre_val in pre_metrics.items():
        post_val = post_metrics.get(key)
        if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)):
            acc.metrics[f"delta_{key}"] = post_val - pre_val


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

    acc = init_suite_metrics(pre_metrics)
    if pre_rows:
        hits = sum(int(r.get("memory_hit", 0)) for r in pre_rows)
        lat_delta = sum(float(r.get("retrieval_latency_ms", 0.0)) for r in pre_rows)
        acc.metrics["memory_hit_rate"] = hits / max(1, len(pre_rows))
        acc.metrics["latency_ms_delta"] = lat_delta / max(1, len(pre_rows))
        rates = [float(r.get("context_match_rate", 0.0)) for r in pre_rows]
        acc.metrics["context_match_rate"] = sum(rates) / max(1, len(rates))
    if post_metrics is not None:
        acc.update_post(post_metrics)
        compute_deltas(acc, pre_metrics, post_metrics)

    metrics: Dict[str, object] = {
        "version": 2,
        "phase": str(getattr(cfg, "mode", "test")),
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "dataset_profile": cfg.get("dataset_profile") or "default",
        "metrics": {cfg.suite: acc.metrics},
        "diagnostics": {cfg.suite: acc.diagnostics},
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
