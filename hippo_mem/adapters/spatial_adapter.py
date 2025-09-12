# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Spatial adapter wrapper providing fusion-compatible interface."""

from __future__ import annotations

from hippo_mem.common import MemoryTokens
from hippo_mem.spatial.adapter import AdapterConfig, SpatialAdapter

from .memory_base import MemoryAdapterBase


class SpatialMemoryAdapter(MemoryAdapterBase):
    """Expose :class:`SpatialAdapter` under the fusion protocol."""

    def __init__(self, cfg: AdapterConfig) -> None:
        inner = SpatialAdapter(cfg)
        super().__init__(inner)

    def hint(self, memory: MemoryTokens | None = None) -> str | None:
        """Return action hint embedded in ``memory`` if present."""

        if memory is None:
            return None
        hint = memory.meta.get("hint")
        return hint if hint else None


__all__ = ["SpatialMemoryAdapter"]
