"""Algorithm Card: SMPD Spatial Adapter

Summary
-------
Cross-attention adapter that fuses plan or macro embeddings with token states.

Integration style
-----------------
Inserted after a transformer block; attends over memory tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from hippo_mem.common.attn_adapter import CrossAttnAdapter, CrossAttnConfig


@dataclass
class AdapterConfig(CrossAttnConfig):
    """Configuration for :class:`SpatialAdapter`."""


class SpatialAdapter(CrossAttnAdapter):
    """Cross-attention module for spatial plans or macros."""


__all__ = ["SpatialAdapter", "AdapterConfig"]
