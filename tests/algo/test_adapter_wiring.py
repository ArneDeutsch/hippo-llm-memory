# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from hippo_mem.adapters import patch
from hippo_mem.adapters.patch import MemoryFusionConfig, attach_adapters
from hippo_mem.common import MemoryTokens
from hippo_mem.testing import FAKE_MODEL_ID


def _setup_model(monkeypatch: pytest.MonkeyPatch) -> AutoModelForCausalLM:
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained(FAKE_MODEL_ID)
    model.eval()
    monkeypatch.setattr(patch, "find_transformer_blocks", lambda m: list(m.transformer.h))
    return model


def test_disabled_memory_matches_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _setup_model(monkeypatch)
    input_ids = torch.tensor([[0]])
    with torch.no_grad():
        logits = model(input_ids).logits[0, 0, :10]

    expected = torch.tensor(
        [
            0.76504016,
            0.06311644,
            -0.09524395,
            -0.08798069,
            0.13141681,
            -0.15304527,
            0.04292956,
            0.12787980,
            -0.13714597,
            0.11853138,
        ]
    )
    assert torch.allclose(logits, expected, rtol=0, atol=1e-5)

    cfg = MemoryFusionConfig(enabled=False)
    attach_adapters(model, cfg)
    with torch.no_grad():
        disabled = model(input_ids).logits[0, 0, :10]
    assert torch.allclose(disabled, expected, rtol=0, atol=1e-5)


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


def test_retrieval_callback_receives_block_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _setup_model(monkeypatch)
    input_ids = torch.tensor([[0]])

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, output):
        captured["baseline"] = output[0] if isinstance(output, tuple) else output

    handle = model.transformer.h[0].register_forward_hook(hook)
    _ = model(input_ids)
    handle.remove()

    cfg = MemoryFusionConfig(enabled=True, insert_block_index=0)

    class Dummy(torch.nn.Module):
        def forward(self, hidden_states, *, memory=None, **kwargs):  # type: ignore[override]
            return torch.zeros_like(hidden_states)

    attach_adapters(model, cfg, episodic=Dummy())

    seen: dict[str, torch.Tensor] = {}

    def cb(hs: torch.Tensor):
        seen["hidden"] = hs
        d = hs.size(-1)
        tokens = hs.new_zeros(hs.size(0), 1, d)
        mask = torch.ones(hs.size(0), 1, dtype=torch.bool, device=hs.device)
        return MemoryTokens(tokens=tokens, mask=mask)

    model._hippo_retrieval_cb = cb  # type: ignore[attr-defined]
    _ = model(input_ids)

    assert "hidden" in seen
    torch.testing.assert_close(seen["hidden"], captured["baseline"])


def test_missing_blocks_attribute_errors() -> None:
    class Dummy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
            return x

    cfg = MemoryFusionConfig()
    with pytest.raises(AttributeError) as exc:
        attach_adapters(Dummy(), cfg)
    assert "Could not find transformer blocks" in str(exc.value)


def test_empty_block_list_errors() -> None:
    class Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers: list[torch.nn.Module] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
            return x

    cfg = MemoryFusionConfig()
    with pytest.raises(ValueError) as exc:
        attach_adapters(Dummy(), cfg)
    assert "empty" in str(exc.value)


def test_remove_hooks_restore_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _setup_model(monkeypatch)
    block = model.transformer.h[0]
    block_forward = block.forward
    model_forward = model.forward
    cfg = MemoryFusionConfig(enabled=True, insert_block_index=0)
    attach_adapters(model, cfg)
    assert block.forward is not block_forward
    assert model.forward is not model_forward
    block._hippo_remove_adapter()  # type: ignore[attr-defined]
    model._hippo_remove_adapter()  # type: ignore[attr-defined]
    assert block.forward.__func__ is block_forward.__func__
    assert model.forward.__func__ is model_forward.__func__
