from __future__ import annotations

"""Priority replay scheduler utilities."""

from typing import List, Tuple, TYPE_CHECKING
import logging
import random

from hippo_mem.episodic.replay import ReplayQueue

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from scripts.train_lora import TrainConfig


logger = logging.getLogger(__name__)


class PriorityReplayScheduler:
    """Schedule interleaved replay batches based on configured mix."""

    def __init__(self, replay_queue: ReplayQueue, mix: "TrainConfig.BatchMix") -> None:
        self.replay_queue = replay_queue
        self.mix = mix
        self.cycle = 0

    def schedule(self, total: int, batch_size: int) -> List[Tuple[str, List[str]]]:
        """Return a shuffled schedule of replay batches.

        Parameters
        ----------
        total:
            Total number of batches to schedule.
        batch_size:
            Number of traces to sample for each episodic replay batch.
        """

        epi = int(self.mix.episodic * total)
        sem = int(self.mix.semantic * total)
        fresh = total - epi - sem

        schedule: List[Tuple[str, List[str]]] = []
        for _ in range(epi):
            ids = self.replay_queue.sample(batch_size)
            schedule.append(("episodic", ids))
        for _ in range(sem):
            schedule.append(("semantic", []))
        for _ in range(fresh):
            schedule.append(("fresh", []))

        random.shuffle(schedule)
        self.cycle += 1

        for idx, (kind, ids) in enumerate(schedule, 1):
            logger.info("cycle %d batch %d: %s %s", self.cycle, idx, kind, ids)

        return schedule
