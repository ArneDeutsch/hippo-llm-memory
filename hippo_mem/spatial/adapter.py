"""Spatial adapter built on :class:`~hippo_mem.common.CrossAttnAdapter`."""

from __future__ import annotations

from dataclasses import dataclass

from hippo_mem.common.attn_adapter import CrossAttnAdapter, CrossAttnConfig


@dataclass
class AdapterConfig(CrossAttnConfig):
    """Configuration for :class:`SpatialAdapter`."""


class SpatialAdapter(CrossAttnAdapter):
    """Cross-attention over plan or macro embeddings."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)


__all__ = ["SpatialAdapter", "AdapterConfig"]
