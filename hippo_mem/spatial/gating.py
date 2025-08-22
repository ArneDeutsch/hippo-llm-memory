"""Gating heuristics for spatial observations.

Summary
-------
Blocks repeated self-observations and rapid flapping between the same
pair of places.  Each observation is scored based on repetition,
recent edge use, and node degree before being written to the
:class:`~hippo_mem.spatial.map.PlaceGraph`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from .map import PlaceGraph


@dataclass
class SpatialGate:
    """Reject degenerate spatial observations."""

    block_threshold: float = 1.0
    repeat_N: int = 3
    recent_window: int = 20
    max_degree: int = 64
    _ctx_hist: Deque[str] = field(init=False)
    _edge_hist: Deque[Tuple[str, str]] = field(init=False)
    _last_ctx: Optional[str] = None

    def __post_init__(self) -> None:
        self._ctx_hist = deque(maxlen=self.repeat_N)
        self._edge_hist = deque(maxlen=self.recent_window)

    def allow(self, context: str, graph: PlaceGraph) -> bool:
        """Return ``True`` if ``context`` should be observed."""

        repeat_pen = 0.0
        if len(self._ctx_hist) >= self.repeat_N - 1 and all(
            c == context for c in list(self._ctx_hist)[-(self.repeat_N - 1) :]
        ):
            repeat_pen = 1.0

        edge_pen = 0.0
        if self._last_ctx is not None and (self._last_ctx, context) in self._edge_hist:
            edge_pen = 1.0

        deg_pen = 0.0
        node_id = graph._context_to_id.get(context)  # noqa: SLF001 - read-only
        if node_id is not None:
            deg = len(graph.graph.get(node_id, {}))
            if deg > self.max_degree:
                deg_pen = (deg - self.max_degree) / self.max_degree

        score = repeat_pen + edge_pen + deg_pen
        allow = score < self.block_threshold

        self._ctx_hist.append(context)
        if allow:
            if self._last_ctx is not None:
                self._edge_hist.append((self._last_ctx, context))
            self._last_ctx = context
        return allow


__all__ = ["SpatialGate"]
