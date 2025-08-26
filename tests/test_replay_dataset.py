from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

from hippo_mem.training.replay_dataset import ReplayIterableDataset


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


def test_replay_dataset_falls_back_when_replay_exhausted() -> None:
    """Exhausted replay iterators yield base items."""

    base_ds = [{"text": "base0"}, {"text": "base1"}]
    replay_iter = iter([{"prompt": "r", "answer": "a"}])

    def replay_reader():
        return replay_iter

    ds = ReplayIterableDataset(base_ds, replay_reader, ratio=1.0, seed=0)
    samples = list(iter(ds))

    assert len(samples) == 2
    assert samples[0]["prompt"] == "r"
    assert samples[1]["text"] == "base1"
    assert "prompt" not in samples[1]


def test_replay_dataset_handles_empty_replay_iter() -> None:
    """Empty replay streams yield only base dataset items."""

    base_ds = [{"text": "base0"}, {"text": "base1"}]

    def replay_reader():
        return iter([])

    ds = ReplayIterableDataset(base_ds, replay_reader, ratio=1.0, seed=0)
    samples = list(iter(ds))

    assert samples == base_ds
    assert all("prompt" not in s for s in samples)
