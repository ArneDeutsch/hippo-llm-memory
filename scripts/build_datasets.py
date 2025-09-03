"""Wrapper around :mod:`hippo_mem.eval.datasets` for CLI use.

Expose dataset profiles including "hard" variants for ``semantic`` and
``episodic_cross`` to avoid baseline saturation.  This script preserves the
original ``scripts`` entry point while routing all functionality to
``hippo_mem.eval.datasets``.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hippo_mem.eval.datasets import *  # noqa: F401,F403
from hippo_mem.eval.datasets import main

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
