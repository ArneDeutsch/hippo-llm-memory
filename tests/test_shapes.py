import pytest
import torch

from hippo_mem.adapters.episodic_adapter import EpisodicMemoryAdapter, EpisodicMemoryConfig
from hippo_mem.common import MemoryTokens


@pytest.mark.parametrize("batch,mem", [(1, 2), (2, 3)])
def test_memory_token_shape(batch: int, mem: int) -> None:
    d_model = 4
    tokens = torch.zeros(batch, mem, d_model)
    mask = torch.ones(batch, mem, dtype=torch.bool)
    mem_tokens = MemoryTokens(tokens=tokens, mask=mask)
    assert mem_tokens.tokens.shape == (batch, mem, d_model)
    assert mem_tokens.mask.shape == (batch, mem)


@pytest.mark.parametrize("batch,seq,mem", [(1, 2, 3), (2, 3, 4)])
def test_mask_broadcasting(batch: int, seq: int, mem: int) -> None:
    cfg = EpisodicMemoryConfig(hidden_size=8, num_heads=2)
    adapter = EpisodicMemoryAdapter(cfg)

    captured: dict[str, torch.Tensor] = {}

    class Dummy(torch.nn.Module):
        def forward(self, hidden_states, memory):
            mask = torch.where(memory.mask, 0.0, float("-inf"))
            captured["mask"] = mask[:, None, :].expand(
                hidden_states.size(0), hidden_states.size(1), mask.size(1)
            )
            return hidden_states

    adapter.inner = Dummy()  # type: ignore[assignment]
    hidden = torch.zeros(batch, seq, cfg.hidden_size)
    tokens = torch.zeros(batch, mem, cfg.hidden_size)
    mask = torch.tensor([[i % 2 == 0 for i in range(mem)] for _ in range(batch)])
    mem_tokens = MemoryTokens(tokens=tokens, mask=mask)
    adapter(hidden, memory=mem_tokens)
    attn_mask = captured["mask"]
    assert attn_mask.shape == (batch, seq, mem)
    expected = torch.where(mask, 0.0, float("-inf"))[:, None, :].expand(batch, seq, mem)
    torch.testing.assert_close(attn_mask, expected)


def test_empty_memory_ok() -> None:
    cfg = EpisodicMemoryConfig(hidden_size=8, num_heads=2)
    adapter = EpisodicMemoryAdapter(cfg)
    hidden = torch.zeros(2, 3, cfg.hidden_size)
    tokens = torch.zeros(2, 0, cfg.hidden_size)
    mask = torch.zeros(2, 0, dtype=torch.bool)
    mem = MemoryTokens(tokens=tokens, mask=mask)
    out = adapter(hidden, memory=mem)
    assert out.shape == hidden.shape
    assert torch.count_nonzero(out) == 0


def test_nonzero_memory_changes_output() -> None:
    """Adapter returns non-zero residual when memory tokens are active."""

    cfg = EpisodicMemoryConfig(hidden_size=8, num_heads=2)
    adapter = EpisodicMemoryAdapter(cfg)
    hidden = torch.zeros(1, 1, cfg.hidden_size)
    tokens = torch.ones(1, 1, cfg.hidden_size)
    mask = torch.ones(1, 1, dtype=torch.bool)
    mem = MemoryTokens(tokens=tokens, mask=mask)
    out = adapter(hidden, memory=mem)
    assert torch.count_nonzero(out) > 0
