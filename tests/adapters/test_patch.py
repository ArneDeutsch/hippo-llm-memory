import torch

from hippo_mem.adapters.patch import AdapterFusion, MemoryFusionConfig
from hippo_mem.common import MemoryTokens


class DummyAdapter(torch.nn.Module):
    def __init__(self, val: float) -> None:
        super().__init__()
        self.val = val

    def forward(self, hidden_states: torch.Tensor, *, memory=None, **kwargs):  # type: ignore[override]
        return torch.full_like(hidden_states, self.val)


def _mem() -> MemoryTokens:
    tokens = torch.zeros(1, 1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    return MemoryTokens(tokens=tokens, mask=mask, meta={})


def test_fusion_toggles() -> None:
    hidden = torch.zeros(1, 2)
    mem = _mem()
    cfg = MemoryFusionConfig(use_episodic=True, use_relational=False, use_spatial=False)
    fusion = AdapterFusion(cfg, DummyAdapter(1.0), DummyAdapter(2.0), DummyAdapter(3.0))
    out = fusion(hidden, memory_tokens=mem)
    assert torch.equal(out, torch.ones_like(hidden))

    cfg.use_episodic = False
    cfg.use_relational = True
    out = fusion(hidden, memory_tokens=mem)
    assert torch.equal(out, torch.full_like(hidden, 2.0))

    cfg.use_relational = False
    cfg.use_spatial = True
    out = fusion(hidden, memory_tokens=mem)
    assert torch.equal(out, torch.full_like(hidden, 3.0))

    cfg.use_episodic = True
    cfg.use_relational = True
    cfg.use_spatial = True
    out = fusion(hidden, memory_tokens=mem)
    assert torch.equal(out, torch.full_like(hidden, 6.0))


def test_fusion_identity_without_memory() -> None:
    hidden = torch.randn(2, 3)
    cfg = MemoryFusionConfig(use_episodic=True, use_relational=True, use_spatial=True)
    fusion = AdapterFusion(cfg, DummyAdapter(1.0), DummyAdapter(2.0), DummyAdapter(3.0))
    out = fusion(hidden, memory_tokens=None)
    assert torch.equal(out, hidden)
