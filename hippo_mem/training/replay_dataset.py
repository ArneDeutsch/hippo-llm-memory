"""Dataset to interleave replay items with base training data."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Iterator

import numpy as np
import torch


class ReplayIterableDataset(torch.utils.data.IterableDataset):
    """Mix base dataset items with replay samples.

    Parameters
    ----------
    base_ds : Iterable
        Underlying dataset yielding dicts with a ``text`` field.
    replay_reader : Callable[[], Iterator[Dict[str, Any]]]
        Function returning an iterator that yields replay records with at least
        ``{"prompt", "answer"}``.
    ratio : float, optional
        Probability of drawing a replay item instead of a base item, by default
        0.3.
    text_key : str, optional
        Key to place formatted ``"{prompt}\nAnswer: {answer}"`` text, by default
        ``"text"``.
    seed : int, optional
        Seed for deterministic sampling, by default 0.
    """

    def __init__(
        self,
        base_ds: Iterable[Dict[str, Any]],
        replay_reader: Callable[[], Iterator[Dict[str, Any]]],
        ratio: float = 0.3,
        text_key: str = "text",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.base_ds = base_ds
        self.replay_reader = replay_reader
        self.ratio = float(ratio)
        self.text_key = text_key
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = np.random.RandomState(self.seed)
        base_iter = iter(self.base_ds)
        replay_iter = self.replay_reader()
        for base_item in base_iter:
            if rng.rand() < self.ratio:
                try:
                    item = next(replay_iter)
                except StopIteration:
                    replay_iter = self.replay_reader()
                    try:
                        item = next(replay_iter)
                    except StopIteration:
                        yield base_item
                        continue
                prompt = item.get("prompt", "")
                answer = item.get("answer", "")
                item[self.text_key] = f"{prompt}\nAnswer: {answer}"
                yield item
            else:
                yield base_item
