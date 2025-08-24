"""Adapters wrapping :mod:`hippo_mem.episodic.adapter` for fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from hippo_mem.common import MemoryTokens
from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter


@dataclass
class EpisodicMemoryConfig(AdapterConfig):
    """Alias of :class:`~hippo_mem.episodic.adapter.AdapterConfig` for export."""


class EpisodicMemoryAdapter(nn.Module):
    """Expose :class:`EpisodicAdapter` under the fusion protocol."""

    def __init__(self, cfg: AdapterConfig, retrieval_dim: int | None = None) -> None:
        super().__init__()
        self.inner = EpisodicAdapter(cfg)
        dim = retrieval_dim or cfg.hidden_size
        self.proj = nn.Linear(dim, cfg.hidden_size)

    def forward(
        self, hidden_states: Tensor, *, memory: MemoryTokens | None = None, **_: Any
    ) -> Tensor:
        """Return the residual produced from ``memory``."""

        if memory is None:
            return torch.zeros_like(hidden_states)
        fused = self.inner(hidden_states, memory)
        return fused - hidden_states


__all__ = ["EpisodicMemoryAdapter", "EpisodicMemoryConfig"]
