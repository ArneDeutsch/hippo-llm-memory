"""Evaluation utilities for hippo_mem.

Expose helper functions for prompt encoding, model configuration, and
evaluation harness utilities used by scripts.
"""

from .encode import encode_prompt
from .harness import (
    EvalConfig,
    Task,
    evaluate,
    evaluate_matrix,
    run_suite,
)
from .models import load_model_config

__all__ = [
    "encode_prompt",
    "load_model_config",
    "EvalConfig",
    "Task",
    "evaluate",
    "evaluate_matrix",
    "run_suite",
]
