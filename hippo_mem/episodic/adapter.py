"""Cross-attention adapter specialising :class:`~hippo_mem.common.CrossAttnAdapter`."""

from __future__ import annotations

from dataclasses import dataclass

from hippo_mem.common.attn_adapter import CrossAttnAdapter, CrossAttnConfig


@dataclass
class AdapterConfig(CrossAttnConfig):
    """Configuration for :class:`EpisodicAdapter`.

    Extends :class:`~hippo_mem.common.attn_adapter.CrossAttnConfig` with the
    ``hopfield`` flag controlling store-side completion.
    """

    hopfield: bool = True


class EpisodicAdapter(CrossAttnAdapter):
    """Cross-attention between hidden states and episodic memory traces."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)


__all__ = ["EpisodicAdapter", "AdapterConfig"]
