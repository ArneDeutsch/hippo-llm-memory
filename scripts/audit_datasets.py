"""CLI wrapper for :mod:`hippo_eval.eval.audit`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hippo_eval.eval.audit import main

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
