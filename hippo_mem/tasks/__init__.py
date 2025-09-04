"""Deprecated shim for :mod:`hippo_mem.tasks`.

Use :mod:`hippo_eval.tasks` instead.
"""

import warnings

from hippo_eval import tasks as _tasks
from hippo_eval.tasks import *  # noqa: F401,F403

warnings.warn(
    "hippo_mem.tasks is deprecated; use hippo_eval.tasks",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = getattr(_tasks, "__all__", [])
__path__ = _tasks.__path__
