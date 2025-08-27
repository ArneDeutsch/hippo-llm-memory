"""CLI wrapper for :mod:`hippo_mem.consolidation.trainer`.

The script exists as a thin entry point so it can be executed without
installing the package.  All implementation lives in the
``hippo_mem.consolidation.trainer`` module.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

from hippo_mem.consolidation.trainer import (  # noqa: E402
    Args,
    load_config,
    main,
    parse_args,
    train,
)

__all__ = ["Args", "load_config", "main", "parse_args", "train"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
