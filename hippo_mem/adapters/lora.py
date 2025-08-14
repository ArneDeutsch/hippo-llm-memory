"""LoRA adapter helpers.

This module provides tiny wrappers around the PEFT library so that other pieces
of the codebase do not need to depend on PEFT directly.  The helpers are small
and deliberately opinionated; they cover the common cases used in the examples
and unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from peft import PeftModel
from transformers import PreTrainedModel


def load_adapter(base_model: PreTrainedModel, adapter_path: Union[str, Path]) -> PeftModel:
    """Load a LoRA adapter from ``adapter_path`` and attach it to ``base_model``."""

    return PeftModel.from_pretrained(base_model, str(adapter_path))


def merge_adapter(model: PeftModel) -> PreTrainedModel:
    """Merge the LoRA weights into the base model and return it.

    The returned model is a standard :class:`~transformers.PreTrainedModel` with
    no remaining PEFT hooks; it can be saved or used for inference as usual.
    """

    return model.merge_and_unload()


def export_adapter(model: PeftModel, output_dir: Union[str, Path]) -> None:
    """Save the adapter weights to ``output_dir``."""

    model.save_pretrained(str(output_dir))
