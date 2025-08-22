"""LoRA adapter helpers.

This module provides tiny wrappers around the PEFT library so that other pieces
of the codebase do not need to depend on PEFT directly.  The helpers are small
and deliberately opinionated; they cover the common cases used in the examples
and unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Protocol, Union

import torch
from peft import PeftModel
from transformers import PreTrainedModel


class TargetModuleStrategy(Protocol):
    """Callable returning target module names for ``model``."""

    def __call__(self, model: PreTrainedModel) -> List[str]: ...


def _targets_gpt2(model: PreTrainedModel) -> List[str]:
    """Targets for GPT-2 style architectures."""

    modules = ["c_attn", "c_proj"]
    if any("c_fc" in name for name, _ in model.named_modules()):
        modules.append("c_fc")
    return modules


def _targets_llama(_: PreTrainedModel) -> List[str]:
    """Targets for LLaMA/Mistral style architectures."""

    return ["q_proj", "k_proj", "v_proj", "o_proj"]


TARGET_MODULE_STRATEGIES: Dict[str, TargetModuleStrategy] = {
    "gpt2": _targets_gpt2,
    "llama": _targets_llama,
    "mistral": _targets_llama,
}


def register_target_module_strategy(model_type: str, strategy: TargetModuleStrategy) -> None:
    """Register ``strategy`` for ``model_type``."""

    # kept: exercised by tests/test_lora_targets.py
    TARGET_MODULE_STRATEGIES[model_type] = strategy


def inspect_first_block(model: PreTrainedModel) -> List[str]:
    """Scan the first transformer block for projection layer names."""

    block = None
    for attr in ("model", "transformer", "base_model"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        if hasattr(sub, "layers") and len(sub.layers) > 0:
            block = sub.layers[0]
            break
        if hasattr(sub, "h") and len(sub.h) > 0:
            block = sub.h[0]
            break
        if (
            hasattr(sub, "decoder")
            and hasattr(sub.decoder, "layers")
            and len(sub.decoder.layers) > 0
        ):
            block = sub.decoder.layers[0]
            break
    if block is None and hasattr(model, "layers") and len(model.layers) > 0:
        block = model.layers[0]
    if block is None:
        return []

    names = set()
    for name, _ in block.named_modules():
        for target in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "c_attn", "c_proj"):
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
