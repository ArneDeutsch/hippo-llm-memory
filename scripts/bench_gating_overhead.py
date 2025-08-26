"""CLI wrapper for :mod:`hippo_mem.episodic.bench`.

Inserts the repository root into ``sys.path`` so the module can be imported
without requiring installation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

# noqa needed since we modify sys.path before import
from hippo_mem.episodic.bench import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
