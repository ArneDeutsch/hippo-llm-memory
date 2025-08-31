"""CLI wrapper for :mod:`hippo_mem.reporting.report` with re-exports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

import hippo_mem.reporting.report as _report  # noqa: E402

globals().update({k: getattr(_report, k) for k in dir(_report) if not k.startswith("__")})

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_report.main())
