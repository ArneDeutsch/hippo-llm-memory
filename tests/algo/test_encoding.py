"""Tests for prompt encoding and decoding helpers."""

from __future__ import annotations

import torch

from hippo_eval.eval import encode_prompt
from hippo_eval.eval.harness import _evaluate
from hippo_eval.eval.types import Task


class DummyChatTokenizer:
    chat_template = "dummy"

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        # return deterministic tensor
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - simple helper
        return " ".join(str(x.item()) for x in ids)


class DummyPlainTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - simple helper
        return " ".join(str(x.item()) for x in ids)


class DummyModel:
    device = torch.device("cpu")

    def generate(self, **inputs):  # pragma: no cover - deterministic output
        inp = inputs["input_ids"]
        gen = torch.tensor([[9, 8]])
        return torch.cat([inp, gen], dim=1)


def test_encode_prompt_uses_chat_template():
    tok = DummyChatTokenizer()
    out = encode_prompt(tok, "hi", torch.device("cpu"))
    assert torch.equal(out["input_ids"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(out["attention_mask"], torch.ones(1, 3, dtype=torch.long))


def test_encode_prompt_falls_back_to_plain_tokenizer():
    tok = DummyPlainTokenizer()
    out = encode_prompt(tok, "hi", torch.device("cpu"))
    assert torch.equal(out["input_ids"], torch.tensor([[4, 5]]))
    assert torch.equal(out["attention_mask"], torch.tensor([[1, 1]]))


def test_generated_slice_excludes_input():
    tok = DummyPlainTokenizer()
    model = DummyModel()
    tasks = [Task(prompt="foo", answer="9 8")]
    rows, _, in_tok, gen_tok, _ = _evaluate(
        tasks,
        modules={},
        tokenizer=tok,
        model=model,
        max_new_tokens=2,
        use_chat_template=False,
        system_prompt=None,
    )
    assert rows[0]["pred"] == "9 8"
    assert in_tok == 2 and gen_tok == 2
