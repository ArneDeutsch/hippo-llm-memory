"""Common gate decision types and interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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


GateResult = GateDecision


@dataclass
class GateCounters:
    """Minimal gate telemetry for metrics."""

    attempts: int = 0
    accepted: int = 0
    skipped: int = 0


class MemoryGate(Protocol):
    """Protocol for gate objects."""

    def decide(self, *args, **kwargs) -> GateDecision:  # pragma: no cover - Protocol
        """Return a :class:`GateDecision`."""
        ...


__all__ = ["GateDecision", "GateResult", "GateCounters", "MemoryGate"]
