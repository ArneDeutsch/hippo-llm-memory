from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import torch
from torch import nn


class Adapter(Protocol):
    """Protocol for memory adapters."""

    def __call__(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...


@dataclass
class MemoryFusionConfig:
    """Configuration for memory adapter fusion."""

    enabled: bool = True
    insert_block_index: int = -1
    use_episodic: bool = True
    use_relational: bool = False
    use_spatial: bool = False


class AdapterFusion(nn.Module):
    """Residually fuses outputs from memory adapters."""

    def __init__(
        self,
        cfg: MemoryFusionConfig,
        episodic: Adapter | None = None,
        relational: Adapter | None = None,
        spatial: Adapter | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.episodic = episodic
        self.relational = relational
        self.spatial = spatial

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.episodic and self.cfg.use_episodic:
            hidden_states = hidden_states + self.episodic(hidden_states, **kwargs)
        if self.relational and self.cfg.use_relational:
            hidden_states = hidden_states + self.relational(hidden_states, **kwargs)
        if self.spatial and self.cfg.use_spatial:
            hidden_states = hidden_states + self.spatial(hidden_states, **kwargs)
        return hidden_states


def find_transformer_blocks(model: nn.Module) -> List[nn.Module]:
    """Return decoder blocks from a transformer model.

    Searches common attributes in order: ``transformer.h``, ``model.layers``, ``layers``.
    """

    candidates = ["transformer.h", "model.layers", "layers"]
    for path in candidates:
        obj: Any = model
        for attr in path.split("."):
            if not hasattr(obj, attr):
                break
            obj = getattr(obj, attr)
        else:
            blocks = obj
            if isinstance(blocks, Sequence) and all(isinstance(b, nn.Module) for b in blocks):
                if len(blocks) == 0:
                    raise ValueError(f"Model attribute '{path}' is empty; expected blocks.")
                return list(blocks)
    raise AttributeError(
        "Could not find transformer blocks; checked 'transformer.h', 'model.layers', and 'layers'."
    )


def attach_adapters(
    model: nn.Module,
    cfg: MemoryFusionConfig,
    episodic: Adapter | None = None,
    relational: Adapter | None = None,
    spatial: Adapter | None = None,
) -> Dict[str, int | None]:
    """Attach memory adapters after a transformer block.

    Returns a dict with the target block index and total number of blocks.
    """

    blocks = find_transformer_blocks(model)
    num_blocks = len(blocks)
    if not cfg.enabled:
        return {"target_block": None, "num_blocks": num_blocks}

    idx = cfg.insert_block_index
    if idx < 0:
        idx = num_blocks + idx
    if idx < 0 or idx >= num_blocks:
        raise IndexError(
            f"insert_block_index {cfg.insert_block_index} out of range for {num_blocks} blocks"
        )
    block = blocks[idx]
    fusion = AdapterFusion(cfg, episodic, relational, spatial)
    orig_forward = block.forward

    def wrapped_forward(*args: Any, **kwargs: Any):
        hidden = orig_forward(*args, **kwargs)
        if isinstance(hidden, tuple):
            hidden_states, *rest = hidden
            hidden_states = fusion(hidden_states, **kwargs)
            return (hidden_states, *rest)
        hidden_states = fusion(hidden, **kwargs)
        return hidden_states

    block.forward = wrapped_forward  # type: ignore[assignment]
    block._hippo_remove_adapter = lambda: setattr(  # type: ignore[attr-defined]
        block, "forward", orig_forward
    )

    return {"target_block": idx, "num_blocks": num_blocks}
