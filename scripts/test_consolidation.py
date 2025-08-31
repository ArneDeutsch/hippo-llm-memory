"""CLI wrapper for :mod:`hippo_mem.consolidation.test_eval` with re-exports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

import hippo_mem.consolidation.test_eval as _mod  # noqa: E402

sys.modules[__name__] = _mod

# ``hippo_mem.consolidation.test_eval.main`` returns a ``dict`` with metrics rather
# than a conventional integer exit status. Previously we forwarded this return
# value to ``SystemExit``, which caused Python to treat the truthy ``dict`` as a
# non‑zero exit code (1). The slow consolidation tests spawn this script via
# ``subprocess.run(check=True)`` and therefore failed despite successful
# execution.  Simply invoke ``main`` and allow a normal zero exit status so that
# callers only fail when an exception is raised.
if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _mod.main()
