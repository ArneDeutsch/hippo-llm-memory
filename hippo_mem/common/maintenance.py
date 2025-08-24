"""Background task manager."""

from __future__ import annotations

import threading
from typing import Callable, Optional


class BackgroundTaskManager:
    """Run periodic maintenance in a background thread.

    Summary
    -------
    Invokes ``tick`` at fixed intervals until stopped. ``tick`` receives a
    ``threading.Event`` that is set when the manager is asked to stop.
    """

    def __init__(self, tick: Callable[[threading.Event], None]) -> None:
        self._tick = tick
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

    def start(self, interval: float) -> None:
        """Start the background loop if not already running."""
        if self._thread is not None:
            return
        stop_event = threading.Event()

        def loop() -> None:
            while not stop_event.wait(interval):
                self._tick(stop_event)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._thread = t
        self._stop_event = stop_event

    def stop(self) -> None:
        """Signal the loop to exit and wait for thread termination."""
        if self._thread is None:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        self._stop_event = None


__all__ = ["BackgroundTaskManager"]
