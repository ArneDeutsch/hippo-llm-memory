"""Cross-attention adapter for spatial plans.

Summary
-------
Provides a small module that fuses LLM hidden states with plan or macro
embeddings using optional LoRA projections and group query attention.

Parameters
----------
None

Returns
-------
None

Raises
------
None

Side Effects
------------
None

Complexity
----------
``O(B * T * H)`` where ``B`` is batch, ``T`` tokens, ``H`` heads.

Examples
--------
>>> cfg = AdapterConfig(hidden_size=4, num_heads=2)
>>> SpatialAdapter(cfg)  # doctest: +ELLIPSIS
<hippo_mem.spatial.adapter.SpatialAdapter object at ...>

See Also
--------
hippo_mem.spatial.map.PlaceGraph
hippo_mem.spatial.macros.MacroLib
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hippo_mem.episodic.adapter import LoraLinear


@dataclass
class AdapterConfig:
    """Configuration for :class:`SpatialAdapter`.

    Summary
    -------
    Defines projection sizes and LoRA settings.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension ``D``.
    num_heads : int
        Number of attention heads ``H``.
    num_kv_heads : int, optional
        K/V heads for GQA; defaults to ``num_heads``.
    lora_r : int, optional
        LoRA rank, by default ``0`` (disabled).
    lora_alpha : int, optional
        LoRA scaling factor, by default ``1``.
    lora_dropout : float, optional
        Dropout rate in ``[0,1]``.
    enabled : bool, optional
        Whether adapter is active.

    Returns
    -------
    None

    Raises
    ------
    None

    Side Effects
    ------------
    None

    Complexity
    ----------
    ``O(1)`` to instantiate.

    Examples
    --------
    >>> AdapterConfig(hidden_size=4, num_heads=2)
    AdapterConfig(hidden_size=4, num_heads=2, num_kv_heads=None, lora_r=0,
                  lora_alpha=1, lora_dropout=0.0, enabled=False)

    See Also
    --------
    SpatialAdapter
    """

    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    enabled: bool = False


class SpatialAdapter(nn.Module):
    """Cross-attention between LLM states and plan/macro embeddings.

    Summary
    -------
    Uses query/key/value projections (optionally LoRA-augmented) to fuse
    token states with plan vectors.
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        """Initialise projections and parameters.

        Parameters
        ----------
        cfg : AdapterConfig
            Adapter configuration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If head sizes do not divide evenly.

        Side Effects
        ------------
        None

        Complexity
        ----------
        ``O(D^2)`` parameter initialisation.

        Examples
        --------
        >>> SpatialAdapter(AdapterConfig(hidden_size=4, num_heads=2))  # doctest: +ELLIPSIS
        <hippo_mem.spatial.adapter.SpatialAdapter object at ...>

        See Also
        --------
        AdapterConfig
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

    # ------------------------------------------------------------------
    def _expand_kv(self, x: Tensor) -> Tensor:
        """Expand K/V heads to match query heads for MQA/GQA.

        Summary
        -------
        Duplicates K/V heads when ``num_kv_heads`` < ``num_heads``.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, K, T, D)`` where ``K`` is ``num_kv_heads``.

        Returns
        -------
        Tensor
            Shape ``(B, H, T, D)`` with expanded heads.

        Raises
        ------
        None

        Side Effects
        ------------
        None

        Complexity
        ----------
        ``O(B * H * T)``.

        Examples
        --------
        >>> cfg = AdapterConfig(hidden_size=4, num_heads=2, num_kv_heads=1)
        >>> ad = SpatialAdapter(cfg)
        >>> x = torch.zeros(1, 1, 1, ad.head_dim)
        >>> ad._expand_kv(x).shape
        torch.Size([1, 2, 1, 2])

        See Also
        --------
        forward
        """

        if self.num_kv_heads == self.num_heads:
            return x
        b, kvh, t, d = x.shape
        if self.num_kv_heads == 1:
            return x.expand(b, self.num_heads, t, d)
        groups = self.num_heads // self.num_kv_heads
        x = x[:, :, None, :, :].expand(b, self.num_kv_heads, groups, t, d)
        return x.reshape(b, self.num_heads, t, d)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: Tensor,
        plans: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse ``hidden_states`` with ``plans`` using cross-attention.

                Summary
                -------
                Projects inputs into ``num_heads`` and combines them via scaled
        dot-product attention.

                Parameters
                ----------
                hidden_states : Tensor
                    Shape ``(B, T_q, D)``.
                plans : Tensor
                    Shape ``(B, T_p, D)``.
                attn_mask : Tensor, optional
                    Additive mask ``(B, T_q, T_p)``.

                Returns
                -------
                Tensor
                    Shape ``(B, T_q, D)`` fused representation.

                Raises
                ------
                None

                Side Effects
                ------------
                Modifies dropout RNG state.

                Complexity
                ----------
                ``O(B * T_q * T_p * D)``.

                Examples
                --------
                >>> cfg = AdapterConfig(hidden_size=4, num_heads=2)
                >>> ad = SpatialAdapter(cfg)
                >>> hs = torch.zeros(1, 1, 4)
                >>> pl = torch.zeros(1, 1, 4)
                >>> ad(hs, pl).shape
                torch.Size([1, 1, 4])

                See Also
                --------
                _expand_kv
        """

        bsz, q_len, _ = hidden_states.shape
        t_len = plans.shape[1]

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(plans).view(bsz, t_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(plans).view(bsz, t_len, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # why: expand K/V for grouped query attention
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


__all__ = ["SpatialAdapter", "AdapterConfig"]
