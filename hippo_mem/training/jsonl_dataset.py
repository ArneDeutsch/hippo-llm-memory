# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import json
from typing import List, Tuple

from datasets import Dataset


def load_jsonl_files(paths: List[str]) -> Dataset:
    """Load JSONL files into a :class:`datasets.Dataset`.

    Each non-empty line in the given files must be a JSON object containing at
    least ``prompt`` and ``answer`` fields. A ``text`` column is generated as
    ``"{prompt}\nAnswer: {answer}"``.
    """
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                answer = obj.get("answer", "")
                rec = {
                    "prompt": prompt,
                    "answer": answer,
                    "text": f"{prompt}\nAnswer: {answer}",
                }
                if "trajectory" in obj:
                    rec["trajectory"] = obj["trajectory"]
                records.append(rec)
    return Dataset.from_list(records)


def split_train_val(train_path: str, val_path: str | None) -> Tuple[Dataset, Dataset]:
    """Load training and validation sets, creating a split if needed.

    If ``val_path`` is ``None`` a deterministic 95/5 split of ``train_path`` is
    returned.
    """
    train_ds = load_jsonl_files([train_path])
    if val_path:
        val_ds = load_jsonl_files([val_path])
        return train_ds, val_ds
    split = train_ds.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]
