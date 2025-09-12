# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
#!/usr/bin/env python3
"""CLI wrapper for :mod:`hippo_eval.eval.baselines` (bench harness)."""

import os

from hippo_eval.eval.baselines import main


def _guard() -> None:
    """Exit early unless ``ALLOW_BENCH=1`` is set."""
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
