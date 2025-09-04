from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hippo_mem.common import MemoryTokens


class MemoryAdapterBase(nn.Module):
    """Shared wrapper logic for memory adapters.

    Subclasses hold the inner adapter in ``self.inner`` and may override
    :meth:`preprocess` to project or enrich memory tokens before fusion.
    The :meth:`forward` surface is stable across adapters and returns the
    residual to add to ``hidden_states``.
    """

    inner: nn.Module

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self, hidden_states: Tensor, *, memory: MemoryTokens | None = None, **_: Any
    ) -> Tensor:
        """Return residual from ``memory`` or zeros when gated off."""

        if memory is None or not torch.any(memory.mask):
            return torch.zeros_like(hidden_states)

        mem = self.preprocess(memory)
        fused = self.inner(hidden_states, memory=mem)
        return fused - hidden_states

    def preprocess(self, memory: MemoryTokens) -> MemoryTokens:
        """Hook for subclasses to transform ``memory`` tokens."""

        return memory

    @staticmethod
    def build_key(hidden_states: Tensor, span: tuple[int, int] | None = None) -> Tensor:
        """Pool ``hidden_states`` and return an L2-normalised key vector."""

        if span is not None:
            start, end = span
            vec = hidden_states[start:end].mean(dim=0)
        else:
            vec = hidden_states[-1]
        return F.normalize(vec, p=2, dim=-1)


__all__ = ["MemoryAdapterBase"]
