"""Asynchronous writer for episodic stores."""

from __future__ import annotations

import queue
import threading
from typing import Union

import numpy as np

from .gating import DGKey
from .store import TraceValue

if False:
    from .store import EpisodicStore  # pragma: no cover


class AsyncWriter:
    """Commit writes to a store on a background thread."""

    def __init__(self, store: "EpisodicStore", maxsize: int = 64) -> None:
        self.store = store
        self.queue: queue.Queue[tuple[Union[np.ndarray, DGKey], TraceValue]] = queue.Queue(maxsize)
        self.stats = {"writes_enqueued": 0, "writes_committed": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, key: Union[np.ndarray, DGKey], value: TraceValue | str) -> None:
        if isinstance(value, str):
            value = TraceValue(provenance=value)
        self.stats["writes_enqueued"] += 1
        self.queue.put((key, value))

    def _worker(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                key, value = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.store.write(key, value)
                self.stats["writes_committed"] += 1
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


__all__ = ["AsyncWriter"]
