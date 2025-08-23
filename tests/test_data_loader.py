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


def test_load_jsonl_files_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines are ignored and text formatting is preserved."""

    file = tmp_path / "sample.jsonl"
    items = [
        {"prompt": "p1  ", "answer": "a1"},
        {"prompt": "p2", "answer": "a2"},
    ]
    with file.open("w", encoding="utf-8") as fh:
        fh.write("\n")
        for item in items:
            fh.write(json.dumps(item) + "\n\n")
        fh.write("   \n")
    ds = jsonl_dataset.load_jsonl_files([str(file)])
    assert ds.num_rows == len(items)
    assert ds[0]["text"] == "p1  \nAnswer: a1"
    assert ds[1]["text"] == "p2\nAnswer: a2"


def test_split_train_val_creates_deterministic_split(tmp_path: Path) -> None:
    """split_train_val yields deterministic 95/5 split without overlap."""

    train_file = tmp_path / "train.jsonl"
    items = [{"prompt": f"p{i}", "answer": f"a{i}"} for i in range(20)]
    with train_file.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    train1, val1 = jsonl_dataset.split_train_val(str(train_file), None)
    train2, val2 = jsonl_dataset.split_train_val(str(train_file), None)
    assert train1.num_rows == 19
    assert val1.num_rows == 1
    train_text1 = {row["text"] for row in train1}
    val_text1 = {row["text"] for row in val1}
    train_text2 = {row["text"] for row in train2}
    val_text2 = {row["text"] for row in val2}
    assert train_text1 == train_text2
    assert val_text1 == val_text2
    assert train_text1.isdisjoint(val_text1)


def test_split_train_val_uses_given_validation_file(tmp_path: Path) -> None:
    """Providing ``val_path`` returns datasets as-is."""

    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_items = [{"prompt": f"tp{i}", "answer": f"ta{i}"} for i in range(3)]
    val_items = [{"prompt": f"vp{i}", "answer": f"va{i}"} for i in range(2)]
    with train_file.open("w", encoding="utf-8") as fh:
        for item in train_items:
            fh.write(json.dumps(item) + "\n")
    with val_file.open("w", encoding="utf-8") as fh:
        for item in val_items:
            fh.write(json.dumps(item) + "\n")
    train_ds, val_ds = jsonl_dataset.split_train_val(str(train_file), str(val_file))
    assert train_ds.num_rows == len(train_items)
    assert val_ds.num_rows == len(val_items)
    assert val_ds[0]["text"] == "vp0\nAnswer: va0"
