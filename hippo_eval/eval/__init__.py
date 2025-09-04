"""Evaluation utilities for hippo_mem.

Expose helper functions for prompt encoding, model configuration, and
evaluation harness utilities used by scripts.
"""

from .encode import encode_prompt
from .harness import (
    Task,
    evaluate,
    evaluate_matrix,
)
from .models import load_model_config

__all__ = [
    "encode_prompt",
    "load_model_config",
    "Task",
    "evaluate",
    "evaluate_matrix",
]
