"""Tiny macro library for spatial trajectories.

`MacroLib` stores successful trajectories and can suggest the top *k*
macros for a desired start and goal.  The scoring mechanism is merely a
stub that favours shorter macros, leaving room for more sophisticated
learned heuristics in future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class Macro:
    """A reusable action sequence."""

    name: str
    trajectory: List[str]
    signature: str
    steps: int
    success_stats: Tuple[int, int]
    last_update: int


class MacroLib:
    """In‑memory store for macros."""

    def __init__(self) -> None:
        self._macros: Dict[str, Macro] = {}
        self._time = 0

    # ------------------------------------------------------------------
    def store(self, name: str, trajectory: Sequence[str]) -> None:
        """Record a new macro by name."""
        traj = list(trajectory)
        sig = f"{traj[0]}->{traj[-1]}" if traj else ""
        self._macros[name] = Macro(
            name,
            traj,
            signature=sig,
            steps=len(traj),
            success_stats=(0, 0),
            last_update=self._time,
        )

    def update_stats(self, macro_name: str, success: bool) -> None:
        """Update success statistics for ``macro_name``."""

        macro = self._macros[macro_name]
        succ, total = macro.success_stats
        total += 1
        if success:
            succ += 1
        macro.success_stats = (succ, total)
        self._time += 1
        macro.last_update = self._time

    def suggest(self, start: str, goal: str, k: int = 1) -> List[Macro]:
        """Return the top *k* macros from *start* to *goal*."""

        candidates: Iterable[Macro] = (
            m
            for m in self._macros.values()
            if m.trajectory and m.trajectory[0] == start and m.trajectory[-1] == goal
        )
        ranked = sorted(candidates, key=self._probability, reverse=True)
        return list(ranked[:k])

    # ------------------------------------------------------------------
    def _probability(self, macro: Macro) -> float:
        """Return the estimated success probability for ``macro``."""

        succ, total = macro.success_stats
        # Laplace smoothing to provide a 0.5 prior for unseen macros
        return (succ + 1) / (total + 2)
