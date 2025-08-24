"""Structured logging for gate decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from .gates import GateDecision


class ProvenanceLogger:
    """Append gate decisions to a line-delimited JSON file."""

    def __init__(self, outdir: str) -> None:
        """Create a logger writing to ``outdir/provenance.ndjson``."""

        self.path = Path(outdir) / "provenance.ndjson"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        mem: str,
        action: str,
        reason: str,
        payload: Dict[str, Any] | None = None,
        score: float | None = None,
    ) -> None:
        """Append a record with ``payload`` and metadata."""

        rec = {
            "ts": time.time(),
            "memory": mem,
            "action": action,
            "reason": reason,
            "score": score,
            "payload": payload or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def log_gate(
    logger: "ProvenanceLogger | None",
    mem: str,
    decision: GateDecision,
    payload: Dict[str, Any],
) -> None:
    """Log ``decision`` to ``logger`` if provided."""

    if logger is None:
        return
    logger.log(
        mem=mem,
        action=decision.action,
        reason=decision.reason,
        score=decision.score,
        payload=payload,
    )
