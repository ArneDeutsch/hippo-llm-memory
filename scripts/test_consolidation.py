"""CLI wrapper for :mod:`hippo_mem.consolidation.test_eval` with re-exports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

import hippo_mem.consolidation.test_eval as _mod  # noqa: E402

sys.modules[__name__] = _mod

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_mod.main())
