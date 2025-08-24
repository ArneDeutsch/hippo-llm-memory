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
    """Configuration options for :class:`EpisodicAdapter`.

    ``hopfield`` toggles modern-Hopfield completion in the accompanying
    :class:`~hippo_mem.episodic.store.EpisodicStore`. The adapter itself ignores
    the flag but training and evaluation scripts consume it to configure the
    store, so it remains part of the public config surface.
    """

    # Whether Hopfield-style completion should be applied during retrieval.
    hopfield: bool = True


class EpisodicAdapter(CrossAttnAdapter):
    """Cross-attention module specialised for episodic traces."""


__all__ = ["EpisodicAdapter", "AdapterConfig"]
