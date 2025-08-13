"""Episodic memory store."""

from typing import List


class EpisodicStore:
    """Placeholder store for episodic memories."""

    def __init__(self) -> None:
        """Create an empty store."""
        self.events: List[str] = []

    def add(self, event: str) -> None:
        """Add an event to the store."""
        self.events.append(event)
