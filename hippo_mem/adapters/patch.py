from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Protocol

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hippo_mem.common import MemoryTokens, TraceSpec


if TYPE_CHECKING:

    class Adapter(Protocol):
        """Protocol for memory adapters."""

        def __call__(
            self,
            hidden_states: torch.Tensor,
            *,
            memory: MemoryTokens | None = None,
            spec: TraceSpec | None = None,
            **kwargs: Any,
        ) -> torch.Tensor: ...

else:  # pragma: no cover - runtime shape agnostic

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_tokens: "MemoryTokens" | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply enabled adapters to ``hidden_states``.

        Adapters are only invoked when ``memory_tokens`` are supplied. This
        preserves baseline model behaviour when memory is absent.
        """

        if memory_tokens is None:
            return hidden_states

        if self.episodic and self.cfg.use_episodic:
            hidden_states = hidden_states + self.episodic(
                hidden_states, memory=memory_tokens, **kwargs
            )
        if self.relational and self.cfg.use_relational:
            hidden_states = hidden_states + self.relational(
                hidden_states, memory=memory_tokens, **kwargs
            )
        if self.spatial and self.cfg.use_spatial:
            hidden_states = hidden_states + self.spatial(
                hidden_states, memory=memory_tokens, **kwargs
            )
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
    orig_block_forward = block.forward

    def wrapped_forward(*args: Any, **kwargs: Any):
        memory_tokens = kwargs.pop("memory_tokens", None)
        hidden = orig_block_forward(*args, **kwargs)
        returned_tuple = isinstance(hidden, tuple)
        if returned_tuple:
            hidden_states, *rest = hidden
        else:
            hidden_states = hidden
            rest = []

        if memory_tokens is None:
            cb = getattr(model, "_hippo_retrieval_cb", None)
            if callable(cb):
                memory_tokens = cb(hidden_states)
            if memory_tokens is None:
                memory_tokens = getattr(model, "_hippo_memory_tokens", None)

        hidden_states = fusion(hidden_states, memory_tokens=memory_tokens, **kwargs)
        if returned_tuple:
            return (hidden_states, *rest)
        return hidden_states

    block.forward = wrapped_forward  # type: ignore[assignment]
    # TODO: appears unused; consider referencing or removing.
    block._hippo_remove_adapter = lambda: setattr(  # type: ignore[attr-defined]
        block, "forward", orig_block_forward
    )

    orig_model_forward = model.forward

    def model_forward(*args: Any, memory_tokens: "MemoryTokens" | None = None, **kwargs: Any):
        model._hippo_memory_tokens = memory_tokens  # type: ignore[attr-defined]
        return orig_model_forward(*args, **kwargs)

    model.forward = model_forward  # type: ignore[assignment]
    # TODO: appears unused; consider referencing or removing.
    model._hippo_remove_adapter = lambda: setattr(  # type: ignore[attr-defined]
        model, "forward", orig_model_forward
    )

    return {"target_block": idx, "num_blocks": num_blocks}
