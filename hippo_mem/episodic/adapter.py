"""Algorithm Card: HEI-NW Episodic Adapter

Summary
-------
Cross-attention adapter that fuses recalled episodic traces into the model.

Integration style
-----------------
Inserted after a transformer block; attends over memory tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from hippo_mem.common.attn_adapter import CrossAttnAdapter, CrossAttnConfig


@dataclass
class AdapterConfig(CrossAttnConfig):
    """Configuration options for :class:`EpisodicAdapter`."""


class EpisodicAdapter(CrossAttnAdapter):
    """Cross-attention module specialised for episodic traces."""


__all__ = ["EpisodicAdapter", "AdapterConfig"]
