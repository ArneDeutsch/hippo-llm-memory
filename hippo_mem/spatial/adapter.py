"""Cross-attention adapter for spatial plans."""

from __future__ import annotations

from dataclasses import dataclass

from hippo_mem.common.attn_adapter import CrossAttnAdapter, LoraLinear


@dataclass
class AdapterConfig:
    """Configuration for :class:`SpatialAdapter`."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    enabled: bool = False


class SpatialAdapter(CrossAttnAdapter):
    """Cross-attention between LLM states and plan embeddings."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(
            cfg.hidden_size,
            cfg.num_heads,
            cfg.num_kv_heads,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        )


__all__ = ["AdapterConfig", "SpatialAdapter", "LoraLinear"]
