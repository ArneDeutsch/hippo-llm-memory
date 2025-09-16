# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from hippo_mem.adapters.lora import (
    BLOCK_EXTRACTORS,
    TARGET_MODULE_STRATEGIES,
    _find_first_block,
    count_trainable_parameters,
    default_target_modules,
    inspect_first_block,
    register_target_module_strategy,
)
from hippo_mem.testing import FAKE_MODEL_ID


def test_default_targets_enable_trainable_params(monkeypatch) -> None:
    """Applying LoRA to default targets yields trainable parameters."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained(FAKE_MODEL_ID)
    targets = default_target_modules(model)
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=2,
        lora_alpha=4,
        target_modules=targets,
        lora_dropout=0.0,
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, config)
    assert count_trainable_parameters(model) > 0


def test_strategy_registration(monkeypatch) -> None:
    """Custom strategies can be registered for new architectures."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained(FAKE_MODEL_ID)
    model.config.model_type = "dummy"
    original = dict(TARGET_MODULE_STRATEGIES)

    def strategy(_: AutoModelForCausalLM) -> list[str]:
        return ["x_proj"]

    register_target_module_strategy("dummy", strategy)
    try:
        assert default_target_modules(model) == ["x_proj"]
    finally:
        TARGET_MODULE_STRATEGIES.clear()
        TARGET_MODULE_STRATEGIES.update(original)


def test_llama_targets_include_mlp(monkeypatch) -> None:
    """LLaMA-style configs hook attention and MLP projections."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained(FAKE_MODEL_ID)
    model.config.model_type = "llama"
    targets = default_target_modules(model)
    assert set(targets) == {"q_proj", "v_proj", "o_proj", "up_proj", "down_proj"}


def test_fallback_decoder_layers() -> None:
    """Models without ``model_type`` use ``decoder.layers`` to infer targets."""

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(2, 2, bias=False)
            self.k_proj = nn.Linear(2, 2, bias=False)
            self.v_proj = nn.Linear(2, 2, bias=False)
            self.o_proj = nn.Linear(2, 2, bias=False)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.decoder = nn.Module()
            self.model.decoder.layers = nn.ModuleList([Block()])
            self.config = SimpleNamespace()

    model = Model()
    expected = ["k_proj", "o_proj", "q_proj", "v_proj"]
    block = _find_first_block(model)
    assert isinstance(block, Block)
    assert inspect_first_block(model) == expected
    assert default_target_modules(model) == expected


def test_block_extractor_dispatch(monkeypatch) -> None:
    """Known model types dispatch via ``BLOCK_EXTRACTORS``."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained(FAKE_MODEL_ID)
    # pre: gpt2 extractor present
    assert "gpt2" in BLOCK_EXTRACTORS
    assert inspect_first_block(model) == ["c_attn", "c_proj"]
