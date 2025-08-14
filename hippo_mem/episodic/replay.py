"""Prioritized replay utilities for episodic memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .gating import novelty


@dataclass
class ReplayItem:
    """Item stored for replay."""

    event: str
    key: np.ndarray
    score: float
    step: int


class PrioritizedReplay:
    """Replay queue mixing gating score, recency and diversity."""

    def __init__(self, maxlen: int = 1024) -> None:
        self.maxlen = maxlen
        self.items: List[ReplayItem] = []
        self.step = 0

    def add(self, event: str, key: np.ndarray, score: float) -> None:
        """Add an event with associated key and gating score."""

        self.step += 1
        item = ReplayItem(
            event=event,
            key=np.asarray(key, dtype="float32"),
            score=float(score),
            step=self.step,
        )
        self.items.append(item)
        if len(self.items) > self.maxlen:
            self.items.pop(0)

    def sample(self, k: int) -> List[str]:
        """Return ``k`` events in priority order."""

        if not self.items:
            return []
        now = self.step
        keys = np.stack([it.key for it in self.items])
        priorities = []
        for i, it in enumerate(self.items):
            recency = 1.0 / (now - it.step + 1)
            if len(self.items) > 1:
                others = np.delete(keys, i, axis=0)
                divers = novelty(it.key, others)
            else:
                divers = 1.0
            p = 0.5 * it.score + 0.3 * recency + 0.2 * divers
            priorities.append(p)
        order = np.argsort(priorities)[::-1]
        return [self.items[i].event for i in order[:k]]
