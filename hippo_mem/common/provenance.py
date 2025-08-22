"""Structured logging for gate decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


class ProvenanceLogger:
    """Append gate decisions to a line-delimited JSON file."""

    def __init__(self, outdir: str) -> None:
        """Create a logger writing to ``outdir/provenance.ndjson``."""

        self.path = Path(outdir) / "provenance.ndjson"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, mem: str, action: str, reason: str, payload: Dict[str, Any]) -> None:
        """Append a record with ``payload`` and metadata."""

        rec = {
            "ts": time.time(),
            "memory": mem,
            "action": action,
            "reason": reason,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
