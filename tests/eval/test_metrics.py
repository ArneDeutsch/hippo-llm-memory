"""Tests for compute metrics helpers."""

from __future__ import annotations

import time

import torch

from hippo_eval.eval.harness import _evaluate
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
        return " ".join(str(x.item()) for x in ids)


def test_time_ms_per_100_calculation(monkeypatch):
    class DummyModel:
        device = torch.device("cpu")

        def generate(self, **inputs):  # pragma: no cover - deterministic output
            inp = inputs["input_ids"]
            gen = torch.tensor([[3, 4]])
            return torch.cat([inp, gen], dim=1)

    tasks = [Task(prompt="p", answer="3 4")]
    # Sequence of perf_counter calls: t0, item_t0, item_t1, t1
    times = iter([0.0, 0.0, 0.02, 0.03])
    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    rows, metrics, in_tok, gen_tok, elapsed = _evaluate(
        tasks,
        modules={},
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        max_new_tokens=2,
        use_chat_template=False,
        system_prompt=None,
    )

    assert in_tok == 2
    assert gen_tok == 2
    assert elapsed == 0.03
    assert rows[0]["pred"] == "3 4"
    assert metrics["em_raw"] == 1.0

    total_tokens = in_tok + gen_tok
    expected = 100 * elapsed * 1000 / max(1, total_tokens)
    assert expected == 750.0


def test_refusal_rate_detection(monkeypatch):
    class DummyModel:
        device = torch.device("cpu")

        def generate(self, **inputs):
            inp = inputs["input_ids"]
            gen = torch.tensor([[5, 6]])
            return torch.cat([inp, gen], dim=1)

    tasks = [Task(prompt="p", answer="x")]

    def fake_decode(ids, skip_special_tokens=True):
        return "I cannot help with that"

    tok = DummyTokenizer()
    monkeypatch.setattr(tok, "decode", fake_decode)
    rows, metrics, *_ = _evaluate(
        tasks,
        modules={},
        tokenizer=tok,
        model=DummyModel(),
        max_new_tokens=2,
        use_chat_template=False,
        system_prompt=None,
    )
    assert rows
    assert metrics["refusal_rate"] == 1.0
