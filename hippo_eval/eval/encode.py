# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
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
    system_prompt: str = "Answer with the exact shortest span from the prompt. No explanations.",
) -> Dict[str, torch.Tensor]:
    """Return tokenized ``prompt`` for ``device``.

    The returned mapping contains both ``input_ids`` and ``attention_mask`` so
    downstream calls can pass an explicit mask to ``model.generate``.  This
    avoids warnings from Transformers when ``pad_token_id`` matches
    ``eos_token_id``.
    """

    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    )
    if use_chat_template and has_chat:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids)
        else:  # pragma: no cover - future-proof for BatchEncoding
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
