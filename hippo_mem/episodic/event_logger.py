"""Lightweight event logging for episodic stores."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List


class EventLogger:
    """Collects maintenance events and counters."""

    def __init__(self, log_file: str | None = None) -> None:
        self.log_file = log_file
        self.events: List[dict[str, Any]] = []
        self.counters: Dict[str, int] = {
            "writes": 0,
            "recalls": 0,
            "hits": 0,
            "requests": 0,
            "maintenance": 0,
        }

    def inc(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def log(self, op: str, info: dict[str, Any] | None = None) -> None:
        event = {"ts": time.time(), "op": op}
        if info:
            event.update(info)
        self.events.append(event)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")

    def status(self) -> dict[str, Any]:
        out = dict(self.counters)
        req = out.get("requests", 0)
        out["hit_rate"] = out["hits"] / req if req else 0.0
        return out


__all__ = ["EventLogger"]
