from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .kg import KnowledgeGraph


class MaintenanceManager:
    """Manage background maintenance tasks for a ``KnowledgeGraph``."""

    def __init__(self, kg: "KnowledgeGraph") -> None:
        self.kg = kg
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

    def start(self, interval: float = 300.0) -> None:
        """Start periodic maintenance in a background thread."""
        if self._bg_thread is not None:
            return
        stop_event = threading.Event()

        def loop() -> None:
            while not stop_event.wait(interval):
                cfg = self.kg.config.get("prune", {})
                self.kg.prune(
                    float(cfg.get("min_conf", 0.0)),
                    cfg.get("max_age"),
                )
                self.kg._log["maintenance"] += 1

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._bg_thread = t
        self._stop_event = stop_event
        self.kg._bg_thread = t
        self.kg._stop_event = stop_event

    def stop(self) -> None:
        """Stop background maintenance thread if running."""
        if self._bg_thread is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        self._bg_thread.join(timeout=1.0)
        self._bg_thread = None
        self._stop_event = None
        self.kg._bg_thread = None
        self.kg._stop_event = None


__all__ = ["MaintenanceManager"]
