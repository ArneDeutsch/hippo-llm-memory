# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Save or merge LoRA adapters.

This script is intentionally tiny; it mirrors the helpers in
:mod:`hippo_mem.adapters.lora` and is mainly exercised by the unit tests.  It
can either save the adapter weights on their own or merge them into the base
model prior to saving.
"""

from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM

from hippo_mem.adapters.lora import export_adapter, load_adapter, merge_adapter
from hippo_mem.testing.fake_hf import resolve_fake_model_id


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Export LoRA adapters")
    parser.add_argument("base_model", help="HuggingFace model name or path")
    parser.add_argument("adapter", help="Path to the trained adapter")
    parser.add_argument("output", help="Directory to save the weights to")
    parser.add_argument(
        "--merge", action="store_true", help="Merge adapter weights into the base model"
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - thin CLI wrapper
    args = parse_args()
    model_name = resolve_fake_model_id(args.base_model) or args.base_model
    base = AutoModelForCausalLM.from_pretrained(model_name)
    model: PeftModel = load_adapter(base, args.adapter)
    if args.merge:
        merged = merge_adapter(model)
        merged.save_pretrained(args.output)
    else:
        export_adapter(model, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
