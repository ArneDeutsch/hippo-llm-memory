from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

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


@given(st.floats(min_value=0.1, max_value=0.9))
def test_replay_dataset_respects_ratio(ratio: float) -> None:
    """Replay items approximate the configured sampling ratio."""

    base_items = [{"text": f"base{i}"} for i in range(50)]
    base_ds = itertools.cycle(base_items)

    replay_records = [{"prompt": "replay", "answer": "a"} for _ in range(5)]

    def replay_reader():
        return iter(replay_records)

    ds = ReplayIterableDataset(base_ds, replay_reader, ratio=ratio, seed=0)
    samples = list(itertools.islice(iter(ds), 1000))

    replay_count = sum("prompt" in item for item in samples)
    sampled_ratio = replay_count / len(samples)

    assert abs(sampled_ratio - ratio) <= 0.05


def test_replay_dataset_falls_back_when_exhausted() -> None:
    """Exhausted replay streams yield base items without errors."""

    base_items = [{"text": f"base{i}"} for i in range(10)]
    base_ds = itertools.cycle(base_items)

    replay_records = [{"prompt": "r", "answer": "a"} for _ in range(2)]

    def replay_reader():
        if getattr(replay_reader, "called", False):
            return iter([])
        replay_reader.called = True
        return iter(replay_records)

    ds = ReplayIterableDataset(base_ds, replay_reader, ratio=1.0, seed=0)
    samples = list(itertools.islice(iter(ds), 6))

    # Only the first two items come from replay; the rest fall back to base data.
    assert [s.get("prompt") for s in samples[:2]] == ["r", "r"]
    assert all("prompt" not in s for s in samples[2:])
