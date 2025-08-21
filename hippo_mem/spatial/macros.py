"""Utility for recording and ranking procedural macros.

Summary
-------
Stores trajectories that solve tasks and ranks them by empirical success rate
for later suggestion.
Side Effects
------------
Macros accumulate in memory; provide ``update_stats`` to age entries.

Complexity
----------
Operations scale with the number of stored macros.

Examples
--------
>>> lib = MacroLib()
>>> lib.store("route", ["a", "b"])
>>> lib.suggest("a", "b")[0].name
'route'

See Also
--------
hippo_mem.spatial.algorithm_card
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class Macro:
    """Reusable action sequence.

    Summary
    -------
    Describes a stored macro with provenance and performance statistics.

    Parameters
    ----------
    name : str
        Macro identifier.
    trajectory : List[str]
        Ordered steps (contexts) composing the macro.
    signature : str
        ``"start->goal"`` representation.
    steps : int
        Number of steps in ``trajectory``.
    success_stats : Tuple[int, int]
        ``(successes, total)`` trials.
    last_update : int
        Monotonic counter for recency tracking.
    Examples
    --------
    >>> Macro("m", ["s", "g"], "s->g", 2, (0, 0), 0)
    Macro(name='m', trajectory=['s', 'g'], signature='s->g', steps=2,
          success_stats=(0, 0), last_update=0)

    See Also
    --------
    MacroLib
    """

    name: str
    trajectory: List[str]
    signature: str
    steps: int  # TODO: appears unused; consider referencing or removing.
    success_stats: Tuple[int, int]
    last_update: int  # TODO: appears unused; consider referencing or removing.


class MacroLib:
    """In-memory macro library.

    Summary
    -------
    Holds ``Macro`` records and suggests those with the highest expected
    success rate.
    """

    def __init__(self) -> None:
        self._macros: Dict[str, Macro] = {}
        self._time = 0

    # ------------------------------------------------------------------
    def store(self, name: str, trajectory: Sequence[str]) -> None:
        """Record a macro.

        Summary
        -------
        Insert a macro into the library with a deterministic start/end
        signature.

        Parameters
        ----------
        name : str
            Macro identifier.
        trajectory : Sequence[str]
            Ordered contexts representing the procedure.
        Side Effects
        ------------
        Updates internal time counter.

        Complexity
        ----------
        ``O(len(trajectory))`` to copy the sequence.

        Examples
        --------
        >>> lib = MacroLib()
        >>> lib.store("m", ["s", "g"])  # doctest: +ELLIPSIS
        >>> lib.suggest("s", "g")[0].name
        'm'

        See Also
        --------
        update_stats
        suggest
        """

        traj = list(trajectory)
        sig = f"{traj[0]}->{traj[-1]}" if traj else ""
        # why: signature provides deterministic lookup key
        self._macros[name] = Macro(
            name,
            traj,
            signature=sig,
            steps=len(traj),
            success_stats=(0, 0),
            last_update=self._time,
        )

    # TODO: appears unused; consider referencing or removing.
    def update_stats(self, macro_name: str, success: bool) -> None:
        """Update macro performance.

        Summary
        -------
        Incorporate the outcome of one macro execution for ranking.

        Parameters
        ----------
        macro_name : str
            Name of the macro to update.
        success : bool
            ``True`` if execution succeeded.
        Raises
        ------
        KeyError
            If ``macro_name`` is unknown.

        Side Effects
        ------------
        Advances ``last_update`` for recency bias.
        Examples
        --------
        >>> lib = MacroLib(); lib.store("m", ["s", "g"])
        >>> lib.update_stats("m", True)
        >>> lib._macros["m"].success_stats
        (1, 1)

        See Also
        --------
        store
        suggest
        """

        macro = self._macros[macro_name]
        succ, total = macro.success_stats
        total += 1
        if success:
            succ += 1
        macro.success_stats = (succ, total)
        self._time += 1
        macro.last_update = self._time  # TODO: appears unused; consider referencing or removing.

    # TODO: appears unused; consider referencing or removing.
    def suggest(self, start: str, goal: str, k: int = 1) -> List[Macro]:
        """Return candidate macros.

        Summary
        -------
        Rank macros from ``start`` to ``goal`` by smoothed success probability
        and return the top ``k``.

        Parameters
        ----------
        start : str
            Start context.
        goal : str
            Goal context.
        k : int, optional
            Number of macros to return, by default ``1``.

        Returns
        -------
        List[Macro]
            Ranked macro list.
        Complexity
        ----------
        ``O(n log n)`` where ``n`` is the number of candidate macros.

        Examples
        --------
        >>> lib = MacroLib()
        >>> lib.store("m1", ["s", "g"])
        >>> lib.store("m2", ["s", "g"])
        >>> lib.update_stats("m2", True)
        >>> [m.name for m in lib.suggest("s", "g", k=2)]
        ['m2', 'm1']

        See Also
        --------
        store
        update_stats
        """

        candidates: Iterable[Macro] = (
            m
            for m in self._macros.values()
            if m.trajectory and m.trajectory[0] == start and m.trajectory[-1] == goal
        )
        ranked = sorted(candidates, key=self._probability, reverse=True)
        return list(ranked[:k])

    # ------------------------------------------------------------------
    def _probability(self, macro: Macro) -> float:
        """Estimate macro success.

        Summary
        -------
        Compute smoothed success probability for ranking.

        Parameters
        ----------
        macro : Macro
            Macro to score.

        Returns
        -------
        float
            Estimated success probability.
        Examples
        --------
        >>> lib = MacroLib(); lib.store("m", ["s", "g"])
        >>> lib._probability(lib._macros["m"])
        0.5

        See Also
        --------
        suggest
        """

        succ, total = macro.success_stats
        # why: Laplace smoothing avoids zero probability for unseen macros
        return (succ + 1) / (total + 2)


__all__ = ["Macro", "MacroLib"]
