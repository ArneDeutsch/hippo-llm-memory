"""Prompt encoding utilities for evaluation harness.

Provides :func:`encode_prompt` which applies chat templates when
available and falls back to direct tokenization otherwise.
"""

from __future__ import annotations

from typing import Dict

import torch


def encode_prompt(
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    use_chat_template: bool = True,
    system_prompt: str = "You are a helpful assistant.",
) -> Dict[str, torch.Tensor]:
    """Return ``input_ids`` for ``prompt`` on ``device``.

    If ``use_chat_template`` is ``True`` and ``tokenizer`` supports
    :func:`~transformers.PreTrainedTokenizer.apply_chat_template` with a defined
    ``chat_template``, the prompt is wrapped in a simple system/user dialogue and
    encoded via ``apply_chat_template``. Otherwise the prompt is tokenized
    directly.
    """

    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    )
    if use_chat_template and has_chat:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return {"input_ids": input_ids.to(device)}
