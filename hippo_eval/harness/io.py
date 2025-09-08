"""Deprecated shim for :mod:`hippo_eval.eval.writers`."""

from __future__ import annotations

import warnings

from hippo_eval.eval.writers import (
    ensure_run_dirs,
    write_baseline_metrics,
    write_csv,
    write_meta,
    write_metrics,
)

warnings.warn(
    "hippo_eval.harness.io is deprecated; import from hippo_eval.eval.writers instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ensure_run_dirs",
    "write_baseline_metrics",
    "write_csv",
    "write_meta",
    "write_metrics",
]
