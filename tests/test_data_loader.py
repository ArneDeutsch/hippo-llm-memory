from __future__ import annotations

import json
from pathlib import Path

from scripts import jsonl_dataset


def test_load_jsonl_files_builds_text(tmp_path: Path) -> None:
    """Loader creates a text field from prompt and answer."""

    file = tmp_path / "sample.jsonl"
    items = [
        {"prompt": "p1", "answer": "a1"},
        {"prompt": "p2", "answer": "a2"},
        {"prompt": "p3", "answer": "a3"},
    ]
    with file.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    ds = jsonl_dataset.load_jsonl_files([str(file)])
    assert ds.num_rows == 3
    assert ds[0]["text"] == "p1\nAnswer: a1"
    assert all("text" in row for row in ds)
