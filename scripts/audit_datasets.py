"""CLI wrapper for :mod:`hippo_eval.eval.audit`.

Adds the repository root to ``sys.path`` so the package can be imported
without installation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

# noqa needed since we modify sys.path before import
from hippo_eval.eval.audit import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
