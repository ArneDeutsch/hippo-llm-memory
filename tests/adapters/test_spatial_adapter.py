import torch

from hippo_mem.adapters.spatial_adapter import SpatialMemoryAdapter
from hippo_mem.common import MemoryTokens
from hippo_mem.spatial.adapter import AdapterConfig


def test_hint_absent_returns_none() -> None:
    cfg = AdapterConfig(hidden_size=4, num_heads=2, enabled=True)
    adapter = SpatialMemoryAdapter(cfg)
    assert adapter.hint(None) is None
    mem = MemoryTokens(
        tokens=torch.zeros(1, 1, 4), mask=torch.ones(1, 1, dtype=torch.bool), meta={}
    )
    assert adapter.hint(mem) is None


def test_hint_reads_metadata() -> None:
    cfg = AdapterConfig(hidden_size=4, num_heads=2, enabled=True)
    adapter = SpatialMemoryAdapter(cfg)
    meta = {"hint": "go", "other": "x"}
    mem = MemoryTokens(
        tokens=torch.zeros(1, 1, 4), mask=torch.ones(1, 1, dtype=torch.bool), meta=meta
    )
    assert adapter.hint(mem) == "go"
