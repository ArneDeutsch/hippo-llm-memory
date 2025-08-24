"""Spatial adapter wrapper providing fusion-compatible interface."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hippo_mem.common import MemoryTokens
from hippo_mem.spatial.adapter import AdapterConfig, SpatialAdapter


class SpatialMemoryAdapter(nn.Module):
    """Expose :class:`SpatialAdapter` under the fusion protocol."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        self.inner = SpatialAdapter(cfg)

    def forward(self, hidden_states: Tensor, *, memory: MemoryTokens | None = None) -> Tensor:
        if memory is None or not torch.any(memory.mask):
            return torch.zeros_like(hidden_states)

        fused = self.inner(hidden_states, memory=memory)
        return fused - hidden_states


__all__ = ["SpatialMemoryAdapter"]
