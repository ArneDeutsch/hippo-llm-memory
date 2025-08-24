"""Cross-attention adapter for spatial plans."""

from __future__ import annotations

from hippo_mem.common.attn_adapter import AdapterConfig, CrossAttnAdapter


class SpatialAdapter(CrossAttnAdapter):
    """Fuse hidden states with plan or macro embeddings."""

    def __init__(self, cfg: AdapterConfig) -> None:  # pragma: no cover - thin wrapper
        super().__init__(cfg)


__all__ = ["SpatialAdapter", "AdapterConfig"]
