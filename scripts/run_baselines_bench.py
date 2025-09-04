#!/usr/bin/env python3
"""CLI wrapper for :mod:`hippo_eval.eval.baselines` (bench harness).

Allows running the module without installing the package by adding the
repository root to ``sys.path`` before importing ``hippo_mem``.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

# noqa needed since we modify sys.path before import
from hippo_eval.eval.baselines import main  # noqa: E402


def _guard() -> None:
    """Exit early unless ``ALLOW_BENCH=1`` is set.

    The bench harness bypasses the full evaluation path and is intended only
    for quick CI smoke tests. Operators should invoke ``scripts/eval_model.py``
    for real runs. Setting ``ALLOW_BENCH=1`` acknowledges this and enables the
    script.
    """

    allow = os.getenv("ALLOW_BENCH")
    if allow != "1":  # pre: only CI smoke should set this
        msg = (
            "scripts/run_baselines_bench.py is CI-only; use scripts/eval_model.py. "
            "Set ALLOW_BENCH=1 to override."
        )
        raise RuntimeError(msg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _guard()
    main()
