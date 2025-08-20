from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from hippo_mem.adapters import patch
from hippo_mem.adapters.patch import MemoryFusionConfig, attach_adapters


def test_adapter_called_once(monkeypatch) -> None:
    """Forward path should invoke attached adapter exactly once."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained("models/tiny-gpt2")
    # monkeypatch block finder for tiny model
    monkeypatch.setattr(patch, "find_transformer_blocks", lambda m: list(m.transformer.h))

    calls = {"n": 0}

    class Dummy(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):  # type: ignore[override]
            calls["n"] += 1
            return hidden_states

    cfg = MemoryFusionConfig(enabled=True, insert_block_index=0)
    attach_adapters(model, cfg, episodic=Dummy())
    input_ids = torch.tensor([[0]])
    model(input_ids)
    assert calls["n"] == 1
