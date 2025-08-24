"""Cross-attention adapter for episodic traces."""

from __future__ import annotations

from hippo_mem.common.attn_adapter import AdapterConfig, CrossAttnAdapter, LoraLinear


class EpisodicAdapter(CrossAttnAdapter):
    """Specialised cross-attention for episodic memory."""

    def __init__(self, cfg: AdapterConfig) -> None:  # pragma: no cover - thin wrapper
        super().__init__(cfg)


__all__ = ["EpisodicAdapter", "AdapterConfig", "LoraLinear"]
