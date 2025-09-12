# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hippo_eval.eval.harness import _evaluate
from hippo_eval.eval.modes import TestStrategy
from hippo_eval.eval.types import Task
from hippo_mem.episodic.types import TraceValue


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, return_tensors):
        return {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - simple helper
        return "Library"


class DummyModel:
    device = torch.device("cpu")

    def generate(self, **inputs):  # pragma: no cover - deterministic output
        inp = inputs["input_ids"]
        gen = torch.tensor([[2]])
        return torch.cat([inp, gen], dim=1)


class DummyStore:
    dim = 4

    def recall(self, query, k, context_key=None):
        tv = TraceValue(
            tokens_span=(0, 4),
            state_sketch=["Carol visited Library"],
            trace_id="t1",
            context_key=context_key,
        )
        trace = type("T", (), {"key": np.ones(self.dim, dtype=np.float32), "value": tv})
        return [trace]

    def to_dense(self, key):  # pragma: no cover - simple stub
        return key


class DummyAdapter:
    def __init__(self) -> None:
        self.proj = nn.Identity()

    def __call__(self, hidden, memory):  # pragma: no cover - simple stub
        return hidden


def test_audit_includes_context():
    tasks = [Task(prompt="Where did Carol go?", answer="Library")]
    modules = {"episodic": {"store": DummyStore(), "adapter": DummyAdapter()}}
    rows, *_ = _evaluate(
        tasks,
        modules=modules,
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        max_new_tokens=1,
        use_chat_template=False,
        system_prompt=None,
        strategy=TestStrategy(),
    )
    row = rows[0]
    assert row["injected_context"] == ["Carol visited Library"]
    assert row["positions"] == [(0, 4)]
    assert row["source"] == ["t1"]
