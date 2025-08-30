"""Wrapper around :mod:`hippo_mem.eval.datasets` with difficulty profiles."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_mem.eval.datasets import main

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
