# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Relational adapter wrapper emitting residuals for fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hippo_mem.common import MemoryTokens
from hippo_mem.relational.adapter import RelationalAdapter


class RelationalMemoryAdapter(nn.Module):
    """Thin wrapper over :class:`~hippo_mem.relational.adapter.RelationalAdapter`."""

    def __init__(self) -> None:
        """Initialise the inner relational adapter."""
        super().__init__()
        self.inner = RelationalAdapter()

    def forward(self, hidden_states: Tensor, *, memory: MemoryTokens | None = None) -> Tensor:
        """
        Fuse KG tokens into ``hidden_states`` via gated cross-attention.

        Parameters
        ----------
        hidden_states:
            Base model activations at the adapter layer.
        memory:
            Packed knowledge-graph token features and their boolean mask.

        Returns
        -------
        Tensor
            Residual to add to ``hidden_states``. When no KG tokens are
            retrieved, a zero tensor is returned so the adapter becomes a
            no-op.

        Notes
        -----
        The wrapper iterates over the batch/sequence and calls
        :class:`RelationalAdapter` to **cross-attend** each query vector to the
        retrieved KG tokens. If ``memory.mask`` indicates that no tokens are
        present, the operation is **gated off** and a zero residual is emitted.
        This Python implementation is CPU bound and meant purely as a
        placeholder until a fused, batched attention kernel is integrated.
        ``memory.meta.get("nodes")`` may list the corresponding KG node names
        for debugging but is otherwise unused by the adapter.
        """
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
