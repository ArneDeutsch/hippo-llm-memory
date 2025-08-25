"""Tests for the chat-aware prompt encoder."""

import torch

from src.eval.encode import encode_prompt


class DummyChatTokenizer:
    chat_template = "dummy"

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        # return deterministic tensor
        return torch.tensor([[1, 2, 3]])


class DummyPlainTokenizer:
    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[4, 5]])}


def test_encode_prompt_uses_chat_template():
    tok = DummyChatTokenizer()
    out = encode_prompt(tok, "hi", torch.device("cpu"))
    assert torch.equal(out["input_ids"], torch.tensor([[1, 2, 3]]))


def test_encode_prompt_falls_back_to_plain_tokenizer():
    tok = DummyPlainTokenizer()
    out = encode_prompt(tok, "hi", torch.device("cpu"))
    assert torch.equal(out["input_ids"], torch.tensor([[4, 5]]))
