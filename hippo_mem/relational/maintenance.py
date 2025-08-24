from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kg import KnowledgeGraph


class MaintenanceManager:
    """Apply prune policies during maintenance ticks."""

    def __init__(self, kg: "KnowledgeGraph") -> None:
        self.kg = kg

    def tick(self, _event: threading.Event) -> None:
        cfg = self.kg.config.get("prune", {})
        self.kg.prune(
            float(cfg.get("min_conf", 0.0)),
            cfg.get("max_age"),
        )
        self.kg._log["maintenance"] += 1


__all__ = ["MaintenanceManager"]
