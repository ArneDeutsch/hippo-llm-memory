from __future__ import annotations

import itertools

from scripts.replay_dataset import ReplayIterableDataset


def test_replay_dataset_interleaves_samples() -> None:
    """ReplayIterableDataset mixes replay items with base data and keeps fields."""

    base_items = [{"text": f"base{i}"} for i in range(50)]
    base_ds = itertools.cycle(base_items)

    replay_records = [{"prompt": "replay", "answer": "a"} for _ in range(5)]

    def replay_reader():
        return iter(replay_records)

    ds = ReplayIterableDataset(base_ds, replay_reader, ratio=0.5, seed=0)
    samples = list(itertools.islice(iter(ds), 1000))

    replay_items = [item for item in samples if "prompt" in item]
    base_sampled = len(samples) - len(replay_items)
    ratio = len(replay_items) / len(samples)

    assert 0.45 <= ratio <= 0.55
    assert base_sampled > 0
    assert all(
        item.get("prompt") == "replay" and item.get("answer") == "a" for item in replay_items
    )
