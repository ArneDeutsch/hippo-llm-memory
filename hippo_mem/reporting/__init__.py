"""Deprecated shim for :mod:`hippo_mem.reporting`.

Use :mod:`hippo_eval.reporting` instead.
"""

import warnings

from hippo_eval import reporting as _reporting
from hippo_eval.reporting import *  # noqa: F401,F403

warnings.warn(
    "hippo_mem.reporting is deprecated; use hippo_eval.reporting",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = getattr(_reporting, "__all__", [])
__path__ = _reporting.__path__
