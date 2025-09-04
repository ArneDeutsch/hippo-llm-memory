"""CLI wrapper for consolidation evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

from hippo_eval.consolidation.eval import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
