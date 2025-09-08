"""Mode definitions and skeleton strategies for evaluation.

Introduces a ``Mode`` enum and minimal strategy implementations carrying
retrieval and ingestion flags. These strategies perform no real work yet and
act as placeholders for future refactors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class Mode(str, Enum):
    """Supported evaluation modes."""

    TEACH = "teach"
    TEST = "test"
    REPLAY = "replay"


class ModeStrategy(Protocol):
    """Skeleton strategy interface for evaluation modes."""

    retrieval_enabled: bool
    ingest_enabled: bool

    def pre_run(self, inputs: Any) -> None:
        """Hook executed before a run starts."""

    def process_task(self, inputs: Any, task: Any) -> Any:
        """Process a single task."""

    def post_run(self, inputs: Any) -> None:
        """Hook executed after a run ends."""


@dataclass
class TeachStrategy:
    """No-op strategy for :class:`Mode.TEACH`."""

    retrieval_enabled: bool = False
    ingest_enabled: bool = True

    def pre_run(self, inputs: Any) -> None:
        return None

    def process_task(self, inputs: Any, task: Any) -> Any:
        return None

    def post_run(self, inputs: Any) -> None:
        return None


@dataclass
class TestStrategy:
    """No-op strategy for :class:`Mode.TEST`."""

    __test__ = False

    retrieval_enabled: bool = True
    ingest_enabled: bool = False

    def pre_run(self, inputs: Any) -> None:
        return None

    def process_task(self, inputs: Any, task: Any) -> Any:
        return None

    def post_run(self, inputs: Any) -> None:
        return None


@dataclass
class ReplayStrategy:
    """No-op strategy for :class:`Mode.REPLAY`."""

    retrieval_enabled: bool = True
    ingest_enabled: bool = False

    def pre_run(self, inputs: Any) -> None:
        return None

    def process_task(self, inputs: Any, task: Any) -> Any:
        return None

    def post_run(self, inputs: Any) -> None:
        return None


def get_mode_strategy(mode: Mode) -> ModeStrategy:
    """Return the skeleton strategy associated with ``mode``."""

    if mode is Mode.TEACH:
        return TeachStrategy()
    if mode is Mode.TEST:
        return TestStrategy()
    if mode is Mode.REPLAY:
        return ReplayStrategy()
    raise ValueError(f"Unsupported mode: {mode!r}")
