"""Spatial map structures."""

from typing import List, Tuple


class SpatialMap:
    """Minimal map representation."""

    def __init__(self) -> None:
        """Initialize an empty map."""
        self.points: List[Tuple[float, float]] = []

    def add_point(self, x: float, y: float) -> None:
        """Add a point to the map."""
        self.points.append((x, y))
