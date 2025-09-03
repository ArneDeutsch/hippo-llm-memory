"""Adapters wrapping :mod:`hippo_mem.episodic.adapter` for fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
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

        if memory is None or not torch.any(memory.mask):
            return torch.zeros_like(hidden_states)

        tokens = self.proj(memory.tokens)
        mem = MemoryTokens(tokens=tokens, mask=memory.mask, meta=memory.meta)
        fused = self.inner(hidden_states, memory=mem)
        return fused - hidden_states

    @staticmethod
    def build_key(hidden_states: Tensor, span: tuple[int, int] | None = None) -> Tensor:
        """Return an L2-normalised key vector from ``hidden_states``.

        Parameters
        ----------
        hidden_states:
            Token-level representations ``[T, H]``.
        span:
            Optional ``(start, end)`` token indices specifying the span to pool.
            When omitted, use the final token.
        """

        if span is not None:
            start, end = span
            vec = hidden_states[start:end].mean(dim=0)
        else:
            vec = hidden_states[-1]
        return F.normalize(vec, p=2, dim=-1)


__all__ = ["EpisodicMemoryAdapter", "EpisodicMemoryConfig"]
