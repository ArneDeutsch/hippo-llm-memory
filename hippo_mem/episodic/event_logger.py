from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


class EventLogger:
    """Track maintenance events and usage counters."""

    def __init__(self, log_file: Optional[str] = None) -> None:
        self.counters: Dict[str, int] = {
            "writes": 0,
            "recalls": 0,
            "hits": 0,
            "requests": 0,
            "maintenance": 0,
        }
        self._events: List[dict[str, Any]] = []
        self._log_file = log_file

    def increment(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + value

    def log_event(self, op: str, info: dict[str, Any]) -> None:
        event = {"ts": time.time(), "op": op, **info}
        self._events.append(event)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                json.dump(event, fh)
                fh.write("\n")
                fh.flush()

    def status(self) -> dict[str, Any]:
        log = dict(self.counters)
        req = log.get("requests", 0)
        log["hit_rate"] = log["hits"] / req if req else 0.0
        return log


__all__ = ["EventLogger"]
