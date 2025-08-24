"""Rollback helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class HistoryEntry:
    """Container for reversible operations."""

    op: str
    data: Any


class RollbackMixin:
    """Mixin providing bounded operation history and rollback."""

    def __init__(self, max_undo: int = 5) -> None:
        self._history: List[HistoryEntry] = []
        self._max_undo = max_undo

    def _push_history(self, op: str, data: Any) -> None:
        self._history.append(HistoryEntry(op, data))
        if len(self._history) > self._max_undo:
            self._history.pop(0)

    def rollback(self, n: int = 1) -> None:
        """Undo the last ``n`` operations."""
        for _ in range(n):
            if not self._history:
                break
            entry = self._history.pop()
            self._apply_rollback(entry)

    def _apply_rollback(self, entry: HistoryEntry) -> None:
        """Restore state for ``entry``.

        Subclasses must implement.
        """
        raise NotImplementedError


__all__ = ["HistoryEntry", "RollbackMixin"]
