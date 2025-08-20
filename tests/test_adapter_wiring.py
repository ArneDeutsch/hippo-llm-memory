from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from hippo_mem.adapters import patch
from hippo_mem.adapters.patch import MemoryFusionConfig, attach_adapters
from hippo_mem.common import MemoryTokens


def _setup_model(monkeypatch: pytest.MonkeyPatch) -> AutoModelForCausalLM:
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained("models/tiny-gpt2")
    model.eval()
    monkeypatch.setattr(patch, "find_transformer_blocks", lambda m: list(m.transformer.h))
    return model


def test_no_memory_tokens_preserves_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _setup_model(monkeypatch)
    input_ids = torch.tensor([[0]])
    baseline = model(input_ids).logits

    calls = {"n": 0}

    class Dummy(torch.nn.Module):
        def forward(self, hidden_states, *, memory=None, **_: dict):  # type: ignore[override]
            calls["n"] += 1
            return torch.ones_like(hidden_states)

    cfg = MemoryFusionConfig(enabled=True, insert_block_index=0)
    attach_adapters(model, cfg, episodic=Dummy())
    out = model(input_ids, memory_tokens=None).logits
    assert calls["n"] == 0
    assert torch.allclose(out, baseline)


def test_adapter_invoked_with_memory_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _setup_model(monkeypatch)
    input_ids = torch.tensor([[0]])
    baseline = model(input_ids).logits

    calls = {"n": 0}

    class Dummy(torch.nn.Module):
        def forward(self, hidden_states, *, memory=None, **_: dict):  # type: ignore[override]
            calls["n"] += 1
            if memory is None:
                return torch.zeros_like(hidden_states)
            val = memory.tokens.sum()
            return hidden_states.new_full(hidden_states.shape, val)

    cfg = MemoryFusionConfig(enabled=True, insert_block_index=0)
    attach_adapters(model, cfg, episodic=Dummy())

    d = model.config.n_embd
    tokens = torch.ones(1, 2, d)
    mask = torch.tensor([[True, False]])
    mem = MemoryTokens(tokens=tokens, mask=mask)
    out = model(input_ids, memory_tokens=mem).logits
    assert calls["n"] == 1
    assert not torch.allclose(out, baseline)
