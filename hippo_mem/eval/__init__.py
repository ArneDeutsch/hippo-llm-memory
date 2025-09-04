"""Deprecated shim for :mod:`hippo_mem.eval`.

Use :mod:`hippo_eval` instead.
"""

import warnings

from hippo_eval import eval as _eval
from hippo_eval.eval import *  # noqa: F401,F403

warnings.warn("hippo_mem.eval is deprecated; use hippo_eval", DeprecationWarning, stacklevel=2)

__all__ = getattr(_eval, "__all__", [])
__path__ = _eval.__path__
