# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Unit tests for the cross-attention adapter."""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from hippo_mem.common import CrossAttnAdapter, CrossAttnConfig, MemoryTokens


def _make_memory(b: int, t: int, d: int) -> MemoryTokens:
    tokens = torch.randn(b, t, d)
    mask = torch.ones(b, t, dtype=torch.bool)
    return MemoryTokens(tokens=tokens, mask=mask)


def test_init_raises_on_invalid_head_sizes() -> None:
    """Configuration errors surface as ValueError."""

    cfg = CrossAttnConfig(hidden_size=10, num_heads=3)
    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        CrossAttnAdapter(cfg)

    cfg = CrossAttnConfig(hidden_size=12, num_heads=4, num_kv_heads=3)
    with pytest.raises(ValueError, match="num_heads must be divisible"):
        CrossAttnAdapter(cfg)


def test_expand_kv_mqa_and_gqa() -> None:
    """KV heads expand correctly for MQA and GQA settings."""

    # MQA: single KV head expands to all query heads
    cfg_mqa = CrossAttnConfig(hidden_size=8, num_heads=4, num_kv_heads=1)
    adapter_mqa = CrossAttnAdapter(cfg_mqa)
    x_mqa = torch.randn(2, 1, 3, 2)
    out_mqa = adapter_mqa._expand_kv(x_mqa)
    assert out_mqa.shape == (2, 4, 3, 2)
    assert torch.allclose(out_mqa, x_mqa.expand(2, 4, 3, 2))

    # GQA: KV heads duplicated in groups
    cfg_gqa = CrossAttnConfig(hidden_size=12, num_heads=6, num_kv_heads=2)
    adapter_gqa = CrossAttnAdapter(cfg_gqa)
    x_gqa = torch.randn(1, 2, 2, 2)
    out_gqa = adapter_gqa._expand_kv(x_gqa)
    assert out_gqa.shape == (1, 6, 2, 2)
    groups = cfg_gqa.num_heads // cfg_gqa.num_kv_heads
    for i in range(cfg_gqa.num_kv_heads):
        start = i * groups
        segment = out_gqa[:, start : start + groups, :, :]
        expected = x_gqa[:, i : i + 1, :, :].expand(1, groups, 2, 2)
        assert torch.allclose(segment, expected)


def test_forward_pass_shapes() -> None:
    """Forward pass returns ``(B, Q, d_model)`` with and without FlashAttention."""

    hidden = torch.randn(1, 3, 8)
    memory = _make_memory(1, 2, 8)

    # Without FlashAttention
    cfg = CrossAttnConfig(hidden_size=8, num_heads=2, flash_attention=False)
    adapter = CrossAttnAdapter(cfg)
    with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        out = adapter(hidden, memory)
    assert out.shape == (1, 3, 8)

    # With FlashAttention
    cfg_flash = CrossAttnConfig(hidden_size=8, num_heads=2, flash_attention=True)
    adapter_flash = CrossAttnAdapter(cfg_flash)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out_flash = adapter_flash(hidden, memory)
    assert out_flash.shape == (1, 3, 8)
