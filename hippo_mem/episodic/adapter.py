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

from dataclasses import dataclass

from hippo_mem.common.attn_adapter import CrossAttnAdapter, LoraLinear


@dataclass
class AdapterConfig:
    """Configuration options for :class:`EpisodicAdapter`."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    enabled: bool = False
    flash_attention: bool = False
    hopfield: bool = True


class EpisodicAdapter(CrossAttnAdapter):
    """Cross-attention over recalled episodic traces."""

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(
            cfg.hidden_size,
            cfg.num_heads,
            cfg.num_kv_heads,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            flash_attention=cfg.flash_attention,
        )


__all__ = ["AdapterConfig", "EpisodicAdapter", "LoraLinear"]
