"""Common gate decision types and interfaces."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass
class GateDecision:
    """Result of a gate decision.

    Parameters
    ----------
    action : str
        Chosen action such as ``"insert"`` or ``"skip"``.
    reason : str
        Human-readable explanation for the decision.
    score : float | None, optional
        Raw gate score when available.
    """

    action: str
    reason: str
    score: float | None = None

    def __iter__(self) -> Iterator[str]:
        """Allow tuple-unpacking for backward compatibility."""

        warnings.warn(
            "GateDecision tuple-unpacking is deprecated; use attributes instead",
            DeprecationWarning,
            stacklevel=2,
        )
        yield self.action
        yield self.reason


GateResult = GateDecision


class MemoryGate(Protocol):
    """Protocol for gate objects."""

    def decide(self, *args, **kwargs) -> GateDecision:  # pragma: no cover - Protocol
        """Return a :class:`GateDecision`."""
        ...
