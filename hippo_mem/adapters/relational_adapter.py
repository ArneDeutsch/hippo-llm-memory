"""Relational adapter wrapper emitting residuals for fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hippo_mem.common import MemoryTokens
from hippo_mem.relational.adapter import RelationalAdapter


class RelationalMemoryAdapter(nn.Module):
    """Thin wrapper over :class:`~hippo_mem.relational.adapter.RelationalAdapter`."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = RelationalAdapter()

    def forward(self, hidden_states: Tensor, *, memory: MemoryTokens | None = None) -> Tensor:
        if memory is None or not torch.any(memory.mask):
            return torch.zeros_like(hidden_states)

        bsz, seq, _ = hidden_states.shape
        out = torch.zeros_like(hidden_states)
        tokens = memory.tokens.detach().cpu().numpy()
        mask = memory.mask.detach().cpu().numpy()
        for b in range(bsz):
            feats = tokens[b][mask[b]]
            if feats.size == 0:
                continue
            for t in range(seq):
                query = hidden_states[b, t].detach().cpu().numpy()
                fused = self.inner(query, feats)
                out[b, t] = torch.from_numpy(fused).to(hidden_states) - hidden_states[b, t]
        return out


__all__ = ["RelationalMemoryAdapter"]
