"""CLI wrapper for :mod:`hippo_mem.training.lora`.

Imports the package after inserting the repository root into ``sys.path`` so
it can be executed without installation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

from hippo_mem.training.lora import (  # noqa: E402
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
