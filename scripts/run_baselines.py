#!/usr/bin/env python3
"""CLI wrapper for :mod:`hippo_mem.eval.baselines`.

Allows running the module without installing the package by adding the
repository root to ``sys.path`` before importing ``hippo_mem``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

# noqa needed since we modify sys.path before import
from hippo_mem.eval.baselines import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
