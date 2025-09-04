"""Legacy wrapper for :mod:`hippo_eval.datasets.cli`.

This small shim ensures the repository root is on ``sys.path`` when invoking the
dataset builder as a standalone script (``python scripts/build_datasets.py``).
``sys.path`` normally points at the ``scripts/`` directory which prevents
``hippo_eval`` from being imported.  Adding the parent directory keeps existing
documentation and tooling working without requiring the package to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# add repository root for ``hippo_eval`` imports when executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - simple path guard
    sys.path.insert(0, str(ROOT))

from hippo_eval.datasets.cli import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
