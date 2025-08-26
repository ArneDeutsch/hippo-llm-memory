"""CLI wrapper for hippo_mem.training.lora."""

from hippo_mem.training.lora import (
    TrainConfig,
    _init_sft_trainer,
    _load_model_and_tokenizer,
    main,
    parse_args,
    train,
)

__all__ = [
    "TrainConfig",
    "_init_sft_trainer",
    "_load_model_and_tokenizer",
    "main",
    "parse_args",
    "train",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
