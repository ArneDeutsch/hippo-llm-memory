"""CLI wrapper for :mod:`hippo_mem.reporting.summarize`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

from hippo_mem.reporting.summarize import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
