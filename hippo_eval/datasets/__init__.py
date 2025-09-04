"""Dataset generation and IO helpers for evaluation suites."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from hippo_eval.tasks.generators import (
    generate_episodic,
    generate_episodic_capacity,
    generate_episodic_cross,
    generate_episodic_multi,
    generate_semantic,
    generate_spatial,
)

from .loaders import iter_split, load_dataset

SIZES = [50, 200, 1000]
SEEDS = [1337, 2025, 4242]

SUITE_TO_GENERATOR = {
    "episodic": generate_episodic,
    "semantic": generate_semantic,
    "spatial": generate_spatial,
    "episodic_multi": generate_episodic_multi,
    "episodic_cross": generate_episodic_cross,
    "episodic_capacity": generate_episodic_capacity,
}


def generate_dataset(
    suite: str, size: int, seed: int, profile: str = "default", **kwargs: object
) -> List[Dict[str, object]]:
    """Dispatch to the generator for ``suite`` with a difficulty profile."""

    try:
        generator = SUITE_TO_GENERATOR[suite]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown suite: {suite}") from exc
    return generator(size, seed, profile=profile, **kwargs)


def write_jsonl(path: Path, items: Iterable[Dict[str, object]]) -> None:
    """Write items to ``path`` in JSON Lines format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj) + "\n")


def sha256_file(path: Path) -> str:
    """Return the SHA256 checksum of ``path``."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def record_checksum(data_path: Path, checksum_file: Path) -> str:
    """Record SHA256 of ``data_path`` in ``checksum_file`` and return it."""

    digest = sha256_file(data_path)
    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, str] = {}
    if checksum_file.exists():
        data = json.loads(checksum_file.read_text())
    data[data_path.name] = digest
    checksum_file.write_text(json.dumps(data, indent=2))
    return digest


def update_dataset_card(
    suite: str,
    suite_dir: Path,
    filename: str,
    digest: str,
    generator_version: str,
) -> None:
    """Update ``dataset_card.json`` for ``suite`` with ``filename`` → ``digest``."""

    card_path = suite_dir / "dataset_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
    else:
        card = {
            "suite": suite,
            "sizes": SIZES,
            "seeds": SEEDS,
            "generator_version": generator_version,
            "files": {},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cli_example": (
                f"python scripts/datasets_cli.py suite={suite} size=<size> seed=<seed> "
                f"out=data/{suite}/<size>_<seed>.jsonl"
            ),
        }
    card["files"][filename] = digest
    card_path.write_text(json.dumps(card, indent=2))


__all__ = [
    "generate_dataset",
    "write_jsonl",
    "sha256_file",
    "record_checksum",
    "update_dataset_card",
    "load_dataset",
    "iter_split",
    "generate_episodic",
    "generate_episodic_multi",
    "generate_episodic_cross",
    "generate_episodic_capacity",
    "generate_semantic",
    "generate_spatial",
]
