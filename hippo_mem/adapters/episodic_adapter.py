"""Adapters wrapping :mod:`hippo_mem.episodic.adapter` for fusion."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from hippo_mem.common import MemoryTokens
from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter

from .memory_base import MemoryAdapterBase


@dataclass
class EpisodicMemoryConfig(AdapterConfig):
    """Alias of :class:`~hippo_mem.episodic.adapter.AdapterConfig` for export."""


class EpisodicMemoryAdapter(MemoryAdapterBase):
    """Expose :class:`EpisodicAdapter` under the fusion protocol."""

    def __init__(self, cfg: AdapterConfig, retrieval_dim: int | None = None) -> None:
        inner = EpisodicAdapter(cfg)
        super().__init__(inner)
        dim = retrieval_dim or cfg.hidden_size
        self.proj = nn.Linear(dim, cfg.hidden_size)

    def preprocess(self, memory: MemoryTokens) -> MemoryTokens:  # type: ignore[override]
        tokens = self.proj(memory.tokens)
        return MemoryTokens(tokens=tokens, mask=memory.mask, meta=memory.meta)


__all__ = ["EpisodicMemoryAdapter", "EpisodicMemoryConfig"]
