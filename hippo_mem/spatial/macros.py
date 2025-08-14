"""Tiny macro library for spatial trajectories.

`MacroLib` stores successful trajectories and can suggest the top *k*
macros for a desired start and goal.  The scoring mechanism is merely a
stub that favours shorter macros, leaving room for more sophisticated
learned heuristics in future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class Macro:
    """A reusable action sequence."""

    name: str
    trajectory: List[str]


class MacroLib:
    """In‑memory store for macros."""

    def __init__(self) -> None:
        self._macros: Dict[str, Macro] = {}

    # ------------------------------------------------------------------
    def store(self, name: str, trajectory: Sequence[str]) -> None:
        """Record a new macro by name."""

        self._macros[name] = Macro(name, list(trajectory))

    def suggest(self, start: str, goal: str, k: int = 1) -> List[Macro]:
        """Return the top *k* macros from *start* to *goal*."""

        candidates: Iterable[Macro] = (
            m
            for m in self._macros.values()
            if m.trajectory and m.trajectory[0] == start and m.trajectory[-1] == goal
        )
        ranked = sorted(candidates, key=self.score, reverse=True)
        return list(ranked[:k])

    # ------------------------------------------------------------------
    def score(self, macro: Macro) -> float:
        """Quality score for a macro (stub).

        The current implementation simply prefers shorter macros by
        returning ``1 / len(trajectory)``.  Real systems could plug in a
        learned value function or success statistics.
        """

        if not macro.trajectory:
            return 0.0
        return 1.0 / len(macro.trajectory)
