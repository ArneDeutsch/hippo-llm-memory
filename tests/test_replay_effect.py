from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hippo_mem.common import MemoryTokens
from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph


def _eval_em(adapter: EpisodicAdapter, dim: int) -> int:
    """Return exact match for a fixed one-token recall task."""
    with torch.no_grad():
        hidden = torch.zeros(1, 1, dim)
        mem = MemoryTokens(tokens=torch.ones(1, 1, dim), mask=torch.ones(1, 1, dtype=torch.bool))
        out = adapter(hidden, mem)
    score = float(out.sum().item())
    return int(score > 0.5)


class _DatasetWorker(ConsolidationWorker):
    """Worker that trains the episodic adapter on a single example."""

    def __init__(
        self, scheduler: ReplayScheduler, model: torch.nn.Module, adapter: EpisodicAdapter, dim: int
    ) -> None:
        super().__init__(scheduler, model, episodic_adapter=adapter, batch_size=1, lr=1.0)
        self.dim = dim

    def _step_episodic(self) -> None:  # pragma: no cover - trivial
        if self.epi_adapter is None:
            return
        hidden = torch.zeros(1, 1, self.dim)
        mem = MemoryTokens(
            tokens=torch.ones(1, 1, self.dim),
            mask=torch.ones(1, 1, dtype=torch.bool),
        )
        out = self.epi_adapter(hidden, mem)
        loss = (1.0 - out).pow(2).mean()
        self._optim_step(loss)


def _em_delta(seed: int, enable_replay: bool) -> int:
    """Return EM delta for a given seed with optional replay."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dim = 2
    store = EpisodicStore(dim)
    kg = KnowledgeGraph()
    scheduler = ReplayScheduler(
        store, kg, batch_mix=SimpleNamespace(episodic=1.0, semantic=0.0, fresh=0.0)
    )
    scheduler.add_trace("t", np.zeros(dim, dtype=np.float32), score=1.0)
    adapter_cfg = AdapterConfig(hidden_size=dim, num_heads=1, lora_r=dim, enabled=True)
    adapter = EpisodicAdapter(adapter_cfg)
    model = torch.nn.Linear(dim, dim)
    pre = _eval_em(adapter, dim)
    if enable_replay:
        worker = _DatasetWorker(scheduler, model, adapter, dim)
        for _ in range(2):
            worker.step_adapters([("episodic", None)])
    post = _eval_em(adapter, dim)
    return post - pre


@pytest.mark.parametrize("seed", [4, 6, 7])
def test_replay_improves_em(seed: int) -> None:
    """Replay should deterministically improve EM across seeds."""
    delta_on = _em_delta(seed, True)
    delta_off = _em_delta(seed, False)
    assert delta_on > 0, f"Expected positive ΔEM with replay for seed {seed}"
    assert delta_off == 0, f"Expected no ΔEM without replay for seed {seed}"
    assert delta_on > delta_off
