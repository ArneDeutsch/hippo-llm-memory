# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Lightweight dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

Example = Dict[str, Any]


def load_dataset(path: str | Path, cfg: Dict[str, Any]) -> List[Example]:
    """Load at most ``cfg['n']`` examples from ``path``."""
    n = int(cfg.get("n", 0))
    items: List[Example] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if n and len(items) >= n:
                break
            if line.strip():
                items.append(json.loads(line))
    return items


def iter_split(ds: Dict[str, List[Example]], split: str) -> Iterator[Example]:
    """Yield examples from ``ds`` keyed by ``split``."""
    for row in ds.get(split, []):
        yield row


__all__ = ["load_dataset", "iter_split"]
