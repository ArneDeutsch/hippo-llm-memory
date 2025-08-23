import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts import jsonl_dataset


def _collect_batches(ds, seed: int) -> list[list[str]]:
    """Return shuffled text batches for ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=2, shuffle=True, generator=generator)
    return [batch["text"] for batch in loader]


def test_repeated_runs_same_seed_yield_identical_batches(tmp_path: Path) -> None:
    """DataLoader produces identical batches when seeded the same."""

    file = tmp_path / "data.jsonl"
    items = [{"prompt": f"p{i}", "answer": f"a{i}"} for i in range(6)]
    with file.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    ds = jsonl_dataset.load_jsonl_files([str(file)])
    run1 = _collect_batches(ds, 42)
    run2 = _collect_batches(ds, 42)
    assert run1 == run2


def test_load_jsonl_files_handles_missing_and_extra_fields(tmp_path: Path) -> None:
    """Loader fills missing fields and ignores extras."""

    file = tmp_path / "data.jsonl"
    items = [
        {"prompt": "p1", "answer": "a1", "extra": "x"},
        {"prompt": "p2"},
        {"answer": "a3"},
    ]
    with file.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    ds = jsonl_dataset.load_jsonl_files([str(file)])
    assert ds.column_names == ["prompt", "answer", "text"]
    assert ds[1]["prompt"] == "p2"
    assert ds[1]["answer"] == ""
    assert ds[1]["text"] == "p2\nAnswer: "
    assert ds[2]["prompt"] == ""
    assert ds[2]["answer"] == "a3"
    assert ds[2]["text"] == "\nAnswer: a3"
