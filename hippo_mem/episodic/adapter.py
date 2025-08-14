"""Cross-attention adapter for episodic memory traces.

This module provides a lightweight implementation of the ``EpisodicAdapter``
layer used throughout the examples.  The adapter performs a single round of
cross-attention between the hidden states of the language model and a set of
recalled memory traces.  It is intentionally small – the goal of the project is
not to provide a drop in replacement for the attention blocks used in large
models but rather to exercise the integration points for memory retrieval.

The layer supports a very small subset of LoRA/QLoRA style low rank updates so
that it can be trained with the rest of the model when desired.  Multi-query and
grouped-query attention are handled by allowing a different number of key/value
heads to query heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoraLinear(nn.Linear):
    """``nn.Linear`` with an optional LoRA adaptation.

    The implementation here is intentionally minimal; it provides just enough
    features for the unit tests and examples.  When ``r`` is zero the layer
    behaves exactly like :class:`~torch.nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.r = r
        if r > 0:
            # LoRA decomposition: W = W + BA
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            self.scaling = lora_alpha / r
            self.lora_dropout = nn.Dropout(lora_dropout)
            # Use standard initialisation for A/B so the adapter starts close to
            # zero but not exactly zero which would block gradients.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 1.0
            self.lora_dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - thin wrapper
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            # (batch, *, in) -> (batch, *, r) -> (batch, *, out)
            lora = F.linear(self.lora_dropout(x), self.lora_A)
            lora = F.linear(lora, self.lora_B)
            result = result + self.scaling * lora
        return result


@dataclass
class AdapterConfig:
    """Configuration options for :class:`EpisodicAdapter`."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    enabled: bool = False


class EpisodicAdapter(nn.Module):
    """Cross-attention over recalled episodic traces."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads or cfg.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = LoraLinear(
            self.hidden_size,
            self.hidden_size,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=False,
        )
        self.k_proj = LoraLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=False,
        )
        self.v_proj = LoraLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=False,
        )
        self.o_proj = LoraLinear(
            self.hidden_size,
            self.hidden_size,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=False,
        )
        self.dropout = nn.Dropout(cfg.lora_dropout)

    # ------------------------------------------------------------------
    def _expand_kv(self, x: Tensor) -> Tensor:
        """Expand K/V heads to match query heads for MQA/GQA."""

        if self.num_kv_heads == self.num_heads:
            return x
        b, kvh, t, d = x.shape
        if self.num_kv_heads == 1:  # multi-query attention
            return x.expand(b, self.num_heads, t, d)
        # grouped-query attention
        groups = self.num_heads // self.num_kv_heads
        x = x[:, :, None, :, :].expand(b, self.num_kv_heads, groups, t, d)
        return x.reshape(b, self.num_heads, t, d)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: Tensor,
        traces: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse ``hidden_states`` with ``traces`` using cross-attention."""

        bsz, q_len, _ = hidden_states.shape
        t_len = traces.shape[1]

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (b, h, q, d)
        k = self.k_proj(traces).view(bsz, t_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(traces).view(bsz, t_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)  # (b, kvh, t, d)
        v = v.transpose(1, 2)
        k = self._expand_kv(k)
        v = self._expand_kv(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask[:, None, :, :]
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(context)
        return hidden_states + out
