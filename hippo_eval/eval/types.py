# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Shared evaluation types.

This module centralizes small dataclasses used across the evaluation pipeline. Keeping them in one place keeps ``harness.py`` slim and avoids circular imports once helpers are split across modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from omegaconf import DictConfig

from hippo_eval.eval.adapters import EvalAdapter
from hippo_mem.common.gates import GateCounters


@dataclass
class Task:
    """Simple container for evaluation items."""

    prompt: str
    answer: str
    qid: str | None = None
    episode_id: str | None = None
    context_key: str | None = None
    fact: str | None = None
    facts: list[dict] | None = None


@dataclass
class RunInputs:
    """Inputs required to execute a suite of tasks."""

    cfg: DictConfig
    modules: Dict[str, Dict[str, object]]
    adapters: Dict[str, EvalAdapter]
    tokenizer: object
    model: object
    gating: Dict[str, GateCounters]
    suite: str | None
    retrieval_enabled: bool
    long_context_enabled: bool
    use_chat_template: bool
    system_prompt: str | None


@dataclass
class RunOutputs:
    """Outputs produced for a single evaluated task."""

    row: Dict[str, object]
    metrics: Dict[str, float]
    in_tokens: int
    gen_tokens: int
    elapsed_s: float
