"""Shared cross-attention adapter utilities."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .specs import MemoryTokens


class LoraLinear(nn.Linear):
    """``nn.Linear`` with optional LoRA weights."""

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
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            self.scaling = lora_alpha / r
            self.lora_dropout = nn.Dropout(lora_dropout)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 1.0
            self.lora_dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - passthrough
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora = F.linear(self.lora_dropout(x), self.lora_A)
            lora = F.linear(lora, self.lora_B)
            result = result + self.scaling * lora
        return result


class CrossAttnAdapter(nn.Module):
    """Multi-head cross-attention over ``MemoryTokens``."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.head_dim = hidden_size // num_heads
        self.q_proj = LoraLinear(
            hidden_size,
            hidden_size,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
        )
        self.k_proj = LoraLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
        )
        self.v_proj = LoraLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
        )
        self.o_proj = LoraLinear(
            hidden_size,
            hidden_size,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
        )
        self.dropout = nn.Dropout(lora_dropout)
        self.use_flash = flash_attention

    def _expand_kv(self, x: Tensor) -> Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        b, _, t, d = x.shape
        if self.num_kv_heads == 1:
            return x.expand(b, self.num_heads, t, d)
        groups = self.num_heads // self.num_kv_heads
        x = x[:, :, None, :, :].expand(b, self.num_kv_heads, groups, t, d)
        return x.reshape(b, self.num_heads, t, d)

    def forward(self, hidden_states: Tensor, memory: MemoryTokens) -> Tensor:
        if memory.tokens.size(1) == 0 or not torch.any(memory.mask):
            return hidden_states
        tokens = memory.tokens
        mask = (~memory.mask).to(hidden_states.dtype) * -1e9
        attn_mask = mask[:, None, None, :]
        bsz, q_len, _ = hidden_states.shape
        t_len = tokens.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = self.k_proj(tokens).view(bsz, t_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(tokens).view(bsz, t_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = self._expand_kv(k)
        v = self._expand_kv(v)
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p, is_causal=False
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + attn_mask
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(context)
        return hidden_states + out


__all__ = ["CrossAttnAdapter", "LoraLinear"]
