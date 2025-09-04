"""Basic dataset loaders for JSONL evaluation files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

Example = Dict[str, object]


def load_dataset(path: str | Path, limit: int | None = None) -> List[Example]:
    """Load up to ``limit`` examples from a JSONL ``path``."""

    p = Path(path)
    items: List[Example] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def iter_split(ds: Iterable[Example], split: str) -> Iterator[Example]:
    """Iterate over ``ds`` returning items for ``split`` (``all`` by default)."""

    if split not in {"train", "test", "all"}:
        raise ValueError(f"Unknown split: {split}")
    return iter(ds)


__all__ = ["load_dataset", "iter_split"]
