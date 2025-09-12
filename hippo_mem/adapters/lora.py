# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""LoRA adapter helpers.

This module provides tiny wrappers around the PEFT library so that other pieces
of the codebase do not need to depend on PEFT directly.  The helpers are small
and deliberately opinionated; they cover the common cases used in the examples
and unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

import torch
from peft import PeftModel
from transformers import PreTrainedModel


class TargetModuleStrategy(Protocol):
    """Callable returning target module names for ``model``."""

    def __call__(self, model: PreTrainedModel) -> List[str]: ...


class BlockExtractor(Protocol):
    """Callable returning the first transformer block for ``model``."""

    def __call__(self, model: PreTrainedModel) -> Optional[torch.nn.Module]: ...


def _targets_gpt2(model: PreTrainedModel) -> List[str]:
    """Targets for GPT-2 style architectures."""

    modules = ["c_attn", "c_proj"]
    if any("c_fc" in name for name, _ in model.named_modules()):
        modules.append("c_fc")
    return modules


def _targets_llama(_: PreTrainedModel) -> List[str]:
    """Targets for LLaMA/Mistral style architectures."""

    # q,v,o from attention and up/down from MLP as default LoRA hooks
    return ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


TARGET_MODULE_STRATEGIES: Dict[str, TargetModuleStrategy] = {
    "gpt2": _targets_gpt2,
    "llama": _targets_llama,
    "mistral": _targets_llama,
}


def _block_gpt2(model: PreTrainedModel) -> Optional[torch.nn.Module]:
    """Return first block for GPT-2 style models."""

    sub = getattr(model, "transformer", None)
    h = getattr(sub, "h", None)
    if h:
        return h[0]
    return None


def _block_llama(model: PreTrainedModel) -> Optional[torch.nn.Module]:
    """Return first block for LLaMA/Mistral models."""

    sub = getattr(model, "model", model)
    layers = getattr(sub, "layers", None)
    if layers:
        return layers[0]
    return None


BLOCK_EXTRACTORS: Dict[str, BlockExtractor] = {
    "gpt2": _block_gpt2,
    "llama": _block_llama,
    "mistral": _block_llama,
}


def _find_first_block(model: PreTrainedModel) -> Optional[torch.nn.Module]:
    """Scan common sub-attributes to locate the first block."""

    for attr in ("model", "transformer", "base_model"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        layers = getattr(sub, "layers", None)
        if layers:
            return layers[0]
        h = getattr(sub, "h", None)
        if h:
            return h[0]
        decoder = getattr(sub, "decoder", None)
        if decoder is not None:
            dec_layers = getattr(decoder, "layers", None)
            if dec_layers:
                return dec_layers[0]
    layers = getattr(model, "layers", None)
    if layers:
        return layers[0]
    return None


def register_target_module_strategy(model_type: str, strategy: TargetModuleStrategy) -> None:
    """Register ``strategy`` for ``model_type``."""

    # kept: exercised by tests/test_lora_targets.py
    TARGET_MODULE_STRATEGIES[model_type] = strategy


def inspect_first_block(model: PreTrainedModel) -> List[str]:
    """Return projection layer names from the first transformer block."""

    model_type = getattr(getattr(model, "config", None), "model_type", "")
    extractor = BLOCK_EXTRACTORS.get(model_type, _find_first_block)
    block = extractor(model)
    if block is None:
        return []

    names: set[str] = set()
    for name, _ in block.named_modules():
        for target in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "qkv",
            "c_attn",
            "c_proj",
        ):
            if name.endswith(target):
                names.add(target)
    return sorted(names)


def load_adapter(
    base_model: PreTrainedModel,
    adapter_path: Union[str, Path],
) -> PeftModel:
    """Load a LoRA adapter from ``adapter_path`` and attach it to
    ``base_model``."""

    return PeftModel.from_pretrained(base_model, str(adapter_path))


def merge_adapter(model: PeftModel) -> PreTrainedModel:
    """Merge the LoRA weights into the base model and return it.

    The returned model is a standard :class:`~transformers.PreTrainedModel`
    with no remaining PEFT hooks; it can be saved or used for inference as
    usual.
    """

    return model.merge_and_unload()


def export_adapter(model: PeftModel, output_dir: Union[str, Path]) -> None:
    """Save the adapter weights to ``output_dir``."""

    model.save_pretrained(str(output_dir))


def default_target_modules(model: PreTrainedModel) -> List[str]:
    """Infer sensible LoRA target modules for ``model``.

    Known architectures dispatch via ``TARGET_MODULE_STRATEGIES``; unknown
    types fall back to :func:`inspect_first_block`.
    """

    model_type = getattr(getattr(model, "config", None), "model_type", "")
    strategy = TARGET_MODULE_STRATEGIES.get(model_type)
    if strategy is not None:
        return strategy(model)
    return inspect_first_block(model)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of parameters with ``requires_grad`` set."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
