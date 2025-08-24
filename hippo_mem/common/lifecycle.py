"""Store lifecycle helpers."""

from __future__ import annotations

import threading
from typing import Optional

from .maintenance import BackgroundTaskManager


class StoreLifecycleMixin:
    """Mixin providing background task management."""

    def __init__(self) -> None:
        self._task_manager: Optional[BackgroundTaskManager] = None
        if not hasattr(self, "_log"):
            self._log = {"writes": 0, "recalls": 0, "hits": 0, "maintenance": 0}

    def log_status(self) -> dict:
        """Return a copy of internal counters."""
        return dict(self._log)

    def start_background_tasks(self, interval: float = 60.0) -> None:
        """Start maintenance thread if not running."""
        if self._task_manager is None:
            self._task_manager = BackgroundTaskManager(self._maintenance_tick)
        self._task_manager.start(interval)

    def stop_background_tasks(self) -> None:
        """Stop maintenance thread if running."""
        if self._task_manager is None:
            return
        self._task_manager.stop()

    def _maintenance_tick(self, event: threading.Event) -> None:
        """Perform maintenance work.

        Subclasses must implement.
        """
        raise NotImplementedError


__all__ = ["StoreLifecycleMixin"]
