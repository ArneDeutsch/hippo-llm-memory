"""Prompt encoding utilities for evaluation harness.

Provides :func:`encode_prompt` which applies chat templates when
available and falls back to direct tokenization otherwise.
"""

from __future__ import annotations

from typing import Dict

import torch


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Return ``input_ids`` for ``prompt`` on ``device``.

    If ``tokenizer`` supports :func:`~transformers.PreTrainedTokenizer.apply_chat_template`
    and has a ``chat_template`` defined, the prompt is wrapped in a default
    system/user conversation and encoded via ``apply_chat_template``.  Otherwise
    the prompt is tokenized directly.
    """

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return {"input_ids": input_ids.to(device)}
