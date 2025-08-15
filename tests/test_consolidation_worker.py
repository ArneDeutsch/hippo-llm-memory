"""Tests for the background consolidation worker."""

import time
from types import SimpleNamespace

import numpy as np
import torch

from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph


class _BatchMix(SimpleNamespace):
    episodic: float = 1.0
    semantic: float = 0.0
    fresh: float = 0.0


def test_worker_updates_adapter() -> None:
    """Worker performs optimisation steps on the episodic adapter."""

    hidden = 4
    store = EpisodicStore(hidden)
    kg = KnowledgeGraph()
    scheduler = ReplayScheduler(store, kg, batch_mix=_BatchMix())
    scheduler.add_trace("t1", np.zeros(hidden, dtype=np.float32), score=1.0)

    model = torch.nn.Linear(hidden, hidden)
    epi_cfg = AdapterConfig(hidden_size=hidden, num_heads=1, lora_r=hidden, enabled=True)
    adapter = EpisodicAdapter(epi_cfg)
    before = adapter.q_proj.lora_B.clone()

    worker = ConsolidationWorker(scheduler, model, episodic_adapter=adapter, batch_size=1)
    worker.start()
    time.sleep(0.1)
    worker.stop()
    worker.join(timeout=1)

    after = adapter.q_proj.lora_B
    assert not torch.equal(before, after), "adapter parameters should update"
