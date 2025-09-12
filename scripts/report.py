# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""CLI wrapper for :mod:`hippo_eval.reporting.report` with re-exports."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import hippo_eval.reporting.report as _report

globals().update({k: getattr(_report, k) for k in dir(_report) if not k.startswith("__")})

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_report.main())
