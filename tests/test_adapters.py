"""Tests for LoRA adapter helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel

from hippo_mem.adapters.lora import export_adapter, load_adapter, merge_adapter
from scripts import export_adapter as export_cli


def _create_adapter(tmp_path: Path) -> Path:
    """Create and save a tiny LoRA adapter for testing."""

    base = AutoModelForCausalLM.from_pretrained("models/tiny-gpt2")
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=2,
        lora_alpha=4,
        target_modules=["c_attn"],
        lora_dropout=0.0,
    )
    model = get_peft_model(base, config)
    out_dir = tmp_path / "adapter"
    model.save_pretrained(out_dir)
    return out_dir


def test_load_merge_export_adapter(tmp_path, monkeypatch) -> None:
    """Helper functions load, export, and merge adapters correctly."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    adapter_dir = _create_adapter(tmp_path)

    base = AutoModelForCausalLM.from_pretrained("models/tiny-gpt2")
    loaded = load_adapter(base, adapter_dir)
    assert isinstance(loaded, PeftModel)

    export_dir = tmp_path / "exported"
    export_adapter(loaded, export_dir)
    assert (export_dir / "adapter_config.json").exists()
    assert any(p.name.startswith("adapter_model") for p in export_dir.iterdir())

    merged = merge_adapter(loaded)
    assert isinstance(merged, PreTrainedModel)
    assert not isinstance(merged, PeftModel)


def test_export_cli_parse_args(monkeypatch) -> None:
    """CLI parses arguments and merge flag."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_adapter", "base", "adapter", "out", "--merge"],
    )
    args = export_cli.parse_args()
    assert args.base_model == "base"
    assert args.adapter == "adapter"
    assert args.output == "out"
    assert args.merge is True
