# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import torch

from hippo_eval.eval.harness import _evaluate
from hippo_eval.eval.modes import TestStrategy
from hippo_eval.eval.types import Task


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, return_tensors):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - simple helper
        return "".join(str(x.item()) for x in ids)


class DummyModel:
    device = torch.device("cpu")

    def generate(self, **inputs):  # pragma: no cover - deterministic output
        inp = inputs["input_ids"]
        gen = torch.tensor([[3, 4]])
        return torch.cat([inp, gen], dim=1)


def test_short_answer_violation(monkeypatch):
    monkeypatch.setenv("HIPPO_ENFORCE_SHORT_ANSWER", "1")
    tok = DummyTokenizer()

    def fake_decode(ids, skip_special_tokens=True):
        return "Library!!!"

    monkeypatch.setattr(tok, "decode", fake_decode)
    tasks = [Task(prompt="p", answer="Library")]
    rows, metrics, *_ = _evaluate(
        tasks,
        modules={},
        tokenizer=tok,
        model=DummyModel(),
        max_new_tokens=2,
        use_chat_template=False,
        system_prompt=None,
        strategy=TestStrategy(),
    )
    assert rows[0]["pred"] == "Library!!!"
    assert rows[0]["normalized_pred"] == ""
    assert rows[0]["format_violation"] == 1
    assert metrics["em_raw"] == 0.0


def test_short_answer_valid(monkeypatch):
    monkeypatch.setenv("HIPPO_ENFORCE_SHORT_ANSWER", "1")
    tok = DummyTokenizer()

    def fake_decode(ids, skip_special_tokens=True):
        return "Library"

    monkeypatch.setattr(tok, "decode", fake_decode)
    tasks = [Task(prompt="p", answer="Library")]
    rows, metrics, *_ = _evaluate(
        tasks,
        modules={},
        tokenizer=tok,
        model=DummyModel(),
        max_new_tokens=2,
        use_chat_template=False,
        system_prompt=None,
        strategy=TestStrategy(),
    )
    assert rows[0]["pred"] == "Library"
    assert rows[0]["normalized_pred"] == "Library"
    assert rows[0]["format_violation"] == 0
    assert metrics["em_raw"] == 1.0
