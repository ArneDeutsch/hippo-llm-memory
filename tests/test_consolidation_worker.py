"""Tests for the background consolidation worker."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic.adapter import AdapterConfig, EpisodicAdapter
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.adapter import AdapterConfig as SpatialAdapterConfig
from hippo_mem.spatial.adapter import SpatialAdapter
from hippo_mem.spatial.map import PlaceGraph


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


def test_worker_updates_spatial_adapter() -> None:
    """Worker also updates the spatial adapter on fresh items."""

    hidden = 4
    store = EpisodicStore(hidden)
    kg = KnowledgeGraph()
    scheduler = ReplayScheduler(
        store, kg, batch_mix=_BatchMix(episodic=0.0, semantic=0.0, fresh=1.0)
    )

    model = torch.nn.Linear(hidden, hidden)
    spat_cfg = SpatialAdapterConfig(hidden_size=hidden, num_heads=1, lora_r=hidden, enabled=True)
    adapter = SpatialAdapter(spat_cfg)
    before = adapter.q_proj.lora_B.clone()

    worker = ConsolidationWorker(scheduler, model, spatial_adapter=adapter, batch_size=1)
    worker.start()
    time.sleep(0.1)
    worker.stop()
    worker.join(timeout=1)

    after = adapter.q_proj.lora_B
    assert not torch.equal(before, after), "spatial adapter parameters should update"


def test_worker_requires_gradients() -> None:
    """Initialisation fails when adapters have no trainable params."""

    hidden = 4
    store = EpisodicStore(hidden)
    kg = KnowledgeGraph()
    scheduler = ReplayScheduler(store, kg, batch_mix=_BatchMix())
    scheduler.add_trace("t1", np.zeros(hidden, dtype=np.float32), score=1.0)

    model = torch.nn.Linear(hidden, hidden)
    cfg = AdapterConfig(hidden_size=hidden, num_heads=1, lora_r=0, enabled=True)
    adapter = EpisodicAdapter(cfg)

    with pytest.raises(ValueError):
        ConsolidationWorker(scheduler, model, episodic_adapter=adapter)


def test_worker_records_maintenance_logs() -> None:
    """Maintenance thread logs events for all stores."""

    hidden = 2
    store = EpisodicStore(hidden, config={"decay_rate": 0.5, "prune": {"min_salience": 0.6}})
    kg = KnowledgeGraph(config={"prune": {"min_conf": 0.6}})
    kg.upsert("A", "rel", "B", "ctx", conf=0.5)
    spat = PlaceGraph(config={"decay_rate": 0.5, "prune": {"max_age": 0}})
    spat.observe("x")
    spat.observe("y")

    scheduler = ReplayScheduler(
        store, kg, batch_mix=_BatchMix(episodic=0.0, semantic=0.0, fresh=1.0)
    )
    worker = ConsolidationWorker(
        scheduler,
        torch.nn.Linear(hidden, hidden),
        episodic_store=store,
        kg_store=kg,
        spatial_map=spat,
        maintenance_interval=0.05,
    )
    worker.start()
    time.sleep(0.2)
    worker.stop()
    worker.join(timeout=1)

    assert any(e["op"] in {"decay", "prune"} for e in store._maintenance_log)
    assert kg._maintenance_log and kg._maintenance_log[-1]["op"] == "prune"
    assert any(e["op"] in {"decay", "prune"} for e in spat._maintenance_log)
