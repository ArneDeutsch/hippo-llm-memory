import json
from collections import Counter
from itertools import islice
from pathlib import Path

import pytest

from hippo_mem.consolidation.replay_dataset import ReplayDataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_replay_sampling_ratios(tmp_path: Path) -> None:
    base = tmp_path / "stores" / "s1"
    epi = [{"prompt": "p", "answer": "a"} for _ in range(5)]
    rel = [{"type": "edge"} for _ in range(5)]
    spa = [{"type": "edge"} for _ in range(5)]
    _write_jsonl(base / "episodic.jsonl", epi)
    _write_jsonl(base / "relational.jsonl", rel)
    _write_jsonl(base / "spatial.jsonl", spa)

    ds = ReplayDataset(
        str(tmp_path / "stores"),
        "s1",
        ratios={"episodic": 0.5, "relational": 0.3, "spatial": 0.2},
        seed=0,
        max_items=100,
        policy="uniform",
    )
    items = list(islice(iter(ds), 100))
    counts = Counter(rec["kind"] for rec in items)
    total = sum(counts.values())
    assert total == 100
    assert counts["episodic"] / total == pytest.approx(0.5, abs=0.1)
    assert counts["relational"] / total == pytest.approx(0.3, abs=0.1)
    assert counts["spatial"] / total == pytest.approx(0.2, abs=0.1)
