"""Deprecated shim for :mod:`hippo_mem.metrics`.

Use :mod:`hippo_eval.metrics` instead.
"""

import warnings

from hippo_eval import metrics as _metrics
from hippo_eval.metrics import *  # noqa: F401,F403

warnings.warn(
    "hippo_mem.metrics is deprecated; use hippo_eval.metrics",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = getattr(_metrics, "__all__", [])
__path__ = _metrics.__path__
