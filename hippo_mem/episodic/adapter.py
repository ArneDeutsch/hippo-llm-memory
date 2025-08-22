"""Algorithm Card: HEI-NW Episodic Adapter

Summary
-------
Cross-attention adapter that fuses recalled episodic traces into the model.

Integration style
-----------------
Inserted after a transformer block; attends over memory tokens.

Data structures
---------------
``DGKey`` sparse keys, ``TraceValue`` payloads, ``AssocStore`` index,
``ReplayQueue`` for consolidation.

Pipeline
--------
1. Encode residual stream → ``DGKey`` via k-WTA.
2. Gate writes using ``S = α·surprise + β·novelty + γ·reward + δ·pin``.
3. Recall traces, optionally complete with Hopfield readout.
4. ``EpisodicAdapter`` cross-attends to fused traces.
5. ReplayScheduler feeds adapter during consolidation.

Design rationale & trade-offs
-----------------------------
Low-rank updates (LoRA) keep adapter lightweight; MQA/GQA limit KV growth but
add projection cost.

Failure modes & diagnostics
---------------------------
Dimension mismatch → verify head counts; empty recalls → check gating and
store.

Ablation switches & expected effects
------------------------------------
``flash_attention=false`` falls back to slower attention; ``hopfield=false``
reduces recall quality.

Contracts
---------
Adapter has no internal state beyond parameters; repeated forwards are
idempotent.
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

    Summary
    -------
    Adds low-rank matrices ``A`` and ``B`` when ``r > 0``.
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
        """Initialise the layer.

        Parameters
        ----------
        in_features : int
            Input feature dimension.
        out_features : int
            Output feature dimension.
        r : int, optional
            Rank of LoRA update; ``0`` disables LoRA.
        lora_alpha : int, optional
            Scaling factor ``α``.
        lora_dropout : float, optional
            Dropout probability for LoRA branch.
        bias : bool, optional
            Include bias term.
        Side Effects
        ------------
        Allocates learnable parameters.

        Complexity
        ----------
        ``O(in_features·out_features)`` for weight initialisation.

        Examples
        --------
        >>> LoraLinear(2, 2, r=1).r
        1

        See Also
        --------
        EpisodicAdapter
        """

        super().__init__(in_features, out_features, bias=bias)
        self.r = r
        if r > 0:
            # why: LoRA decomposition W = W + BA
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            self.scaling = lora_alpha / r
            self.lora_dropout = nn.Dropout(lora_dropout)
            # why: initialise near zero yet allow gradients
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 1.0
            self.lora_dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - thin wrapper
        """Apply linear layer with optional LoRA branch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor ``(..., out_features)``.
        Complexity
        ----------
        ``O(in_features·out_features)``.

        Examples
        --------
        >>> layer = LoraLinear(2, 2, r=1)
        >>> layer(torch.zeros(1,2)).shape
        torch.Size([1, 2])

        See Also
        --------
        __init__
        """

        result = F.linear(x, self.weight, self.bias)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            # (batch, *, in) -> (batch, *, r) -> (batch, *, out)
            lora = F.linear(self.lora_dropout(x), self.lora_A)
            lora = F.linear(lora, self.lora_B)
            result = result + self.scaling * lora
        return result


@dataclass
class AdapterConfig:
    """Configuration options for :class:`EpisodicAdapter`.

    Summary
    -------
    Holds architectural and LoRA hyper-parameters.

    Parameters
    ----------
    hidden_size : int
        Model dimensionality ``H``.
    num_heads : int
        Number of query heads.
    num_kv_heads : int, optional
        Number of key/value heads.
    lora_r : int, optional
        LoRA rank.
    lora_alpha : int, optional
        LoRA scaling factor.
    lora_dropout : float, optional
        Dropout probability for LoRA.
    enabled : bool, optional
        Whether adapter is active.
    flash_attention : bool, optional
        Use fused flash attention if available.
    hopfield : bool, optional
        Enable Hopfield completion in store.
    Examples
    --------
    >>> AdapterConfig(hidden_size=4, num_heads=1).enabled
    False

    See Also
    --------
    EpisodicAdapter
    """

    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    enabled: bool = False
    flash_attention: bool = False
    # Whether Hopfield-style completion should be used. This flag is
    # consumed by the training script to configure the accompanying
    # ``EpisodicStore`` and is ignored by the adapter itself.
    hopfield: bool = True


class EpisodicAdapter(nn.Module):
    """Cross-attention over recalled episodic traces.

    Summary
    -------
    Performs multi-head attention between hidden states and memory traces.
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        """Initialise projection layers.

        Parameters
        ----------
        cfg : AdapterConfig
            Configuration for sizes and LoRA options.
        Raises
        ------
        ValueError
            If head dimensions are incompatible.

        Side Effects
        ------------
        Allocates projection layers.

        Complexity
        ----------
        ``O(hidden_size²)`` for weight allocation.

        Examples
        --------
        >>> cfg = AdapterConfig(hidden_size=4, num_heads=1)
        >>> EpisodicAdapter(cfg).head_dim
        4

        See Also
        --------
        forward
        """

        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads or cfg.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
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
        self.use_flash = cfg.flash_attention

    # ------------------------------------------------------------------
    def _expand_kv(self, x: Tensor) -> Tensor:
        """Expand K/V heads to match query heads for MQA/GQA.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with shape ``(B, kv_heads, T, D)``.

        Returns
        -------
        torch.Tensor
            Expanded tensor ``(B, heads, T, D)``.
        Complexity
        ----------
        ``O(B·T·D)``.

        Examples
        --------
        >>> cfg = AdapterConfig(hidden_size=4, num_heads=2, num_kv_heads=1)
        >>> adapter = EpisodicAdapter(cfg)
        >>> adapter._expand_kv(torch.zeros(1,1,1,2)).shape
        torch.Size([1, 2, 1, 2])

        See Also
        --------
        forward
        """

        if self.num_kv_heads == self.num_heads:
            return x
        b, _, t, d = x.shape  # kvh unused by design
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
        """Fuse ``hidden_states`` with ``traces`` using cross-attention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Query states ``(B, Q, H)``.
        traces : torch.Tensor
            Memory traces ``(B, T, H)``.
        attn_mask : torch.Tensor, optional
            Attention mask ``(B, Q, T)``.

        Returns
        -------
        torch.Tensor
            Fused states ``(B, Q, H)``.
        Complexity
        ----------
        ``O(B·Q·T·H)``.

        Examples
        --------
        >>> cfg = AdapterConfig(hidden_size=4, num_heads=1)
        >>> adapter = EpisodicAdapter(cfg)
        >>> hs = torch.zeros(1,1,4)
        >>> tr = torch.zeros(1,1,4)
        >>> adapter(hs, tr).shape
        torch.Size([1, 1, 4])

        See Also
        --------
        _expand_kv
        """

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
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            mask = None
            if attn_mask is not None:
                mask = attn_mask[:, None, :, :]
            # why: use fused kernel for speed when available
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout.p,
                is_causal=False,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask[:, None, :, :]
            attn = F.softmax(attn_scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(context)
        return hidden_states + out
