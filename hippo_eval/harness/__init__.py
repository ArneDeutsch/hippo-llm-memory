"""Evaluation harness utilities."""

from hippo_eval.harness.io import (
    ensure_run_dirs,
    write_baseline_metrics,
    write_csv,
    write_meta,
    write_metrics,
)
from hippo_eval.harness.metrics import MetricSchema, collect_metrics
from hippo_eval.harness.runner import Runner, RunResult, build_runner, run_suite


def evaluate(*args, **kwargs):
    """Proxy to :func:`hippo_eval.eval.harness.evaluate` to avoid circular imports."""

    from hippo_eval.eval.harness import evaluate as _evaluate

    return _evaluate(*args, **kwargs)


__all__ = [
    "evaluate",
    "ensure_run_dirs",
    "write_baseline_metrics",
    "write_csv",
    "write_meta",
    "write_metrics",
    "MetricSchema",
    "collect_metrics",
    "RunResult",
    "Runner",
    "build_runner",
    "run_suite",
]
