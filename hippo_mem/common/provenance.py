"""Minimal provenance logger for gate decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ProvenanceLogger:
    """Append gate decisions to an ndjson file."""

    def __init__(self, outdir: str) -> None:
        self.path = Path(outdir) / "provenance.ndjson"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, mem: str, action: str, reason: str, payload: dict[str, Any]) -> None:
        rec = {"ts": time.time(), "memory": mem, "action": action, "reason": reason, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
