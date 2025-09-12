# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Wrapper for :mod:`hippo_eval.datasets.cli`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hippo_eval.datasets.cli import main

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
