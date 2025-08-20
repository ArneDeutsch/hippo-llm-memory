from __future__ import annotations

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from hippo_mem.adapters.lora import (
    count_trainable_parameters,
    default_target_modules,
)


def test_default_targets_enable_trainable_params(monkeypatch) -> None:
    """Applying LoRA to default targets yields trainable parameters."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained("models/tiny-gpt2")
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
