import importlib

import pytest
import torch

from hippo_mem.common import TraceSpec


def test_memory_tokens_shape_validation(monkeypatch):
    monkeypatch.setenv("HIPPO_DEBUG", "1")
    mod = importlib.reload(importlib.import_module("hippo_mem.common.specs"))
    tokens = torch.zeros(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    mem = mod.MemoryTokens(tokens=tokens, mask=mask)
    assert mem.tokens.shape == tokens.shape
    assert mem.mask.shape == mask.shape
    bad_mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="dimensions must align"):
        mod.MemoryTokens(tokens=tokens, mask=bad_mask)


def test_trace_spec_fields():
    spec = TraceSpec(source="episodic", k=5, max_len=10)
    assert spec.source == "episodic"
    assert spec.k == 5
    assert spec.max_len == 10
