"""Tests for the background consolidation worker."""

import logging
import threading
import time
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch

from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
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


def test_worker_skips_without_trainable_params() -> None:
    """Worker initialises and runs when adapters lack trainable params."""

    hidden = 4
    store = EpisodicStore(hidden)
    kg = KnowledgeGraph()
    scheduler = ReplayScheduler(store, kg, batch_mix=_BatchMix())
    scheduler.add_trace("t1", np.zeros(hidden, dtype=np.float32), score=1.0)

    model = torch.nn.Linear(hidden, hidden)
    cfg = AdapterConfig(hidden_size=hidden, num_heads=1, lora_r=0, enabled=True)
    adapter = EpisodicAdapter(cfg)

    worker = ConsolidationWorker(scheduler, model, episodic_adapter=adapter, batch_size=1)
    assert worker.optimizer is None, "no optimiser when no trainable params"
    worker.start()
    time.sleep(0.05)
    worker.stop()
    worker.join(timeout=1)


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


def test_setup_scheduler_freezes_non_lora() -> None:
    """`_setup_scheduler` freezes base and adapter params except LoRA layers."""

    hidden = 4
    store = EpisodicStore(hidden)
    kg = KnowledgeGraph()
    scheduler = ReplayScheduler(store, kg, batch_mix=_BatchMix())
    model = torch.nn.Linear(hidden, hidden)
    cfg = AdapterConfig(hidden_size=hidden, num_heads=1, lora_r=hidden, enabled=True)
    adapter = EpisodicAdapter(cfg)

    worker = ConsolidationWorker(scheduler, model)
    worker._setup_scheduler(scheduler, model, adapter, None, None, lr=1e-4)

    assert all(not p.requires_grad for n, p in adapter.named_parameters() if "lora_" not in n)
    assert all(p.requires_grad for n, p in adapter.named_parameters() if "lora_" in n)


def test_setup_maintenance_sets_state() -> None:
    """`_setup_maintenance` stores references and creates event."""

    hidden = 2
    store = EpisodicStore(hidden)
    kg = KnowledgeGraph()
    spat = PlaceGraph()
    scheduler = ReplayScheduler(store, kg, batch_mix=_BatchMix())
    worker = ConsolidationWorker(scheduler, torch.nn.Linear(hidden, hidden))
    worker._setup_maintenance(store, kg, spat, 0.5)

    assert worker.epi_store is store and worker.kg_store is kg
    assert worker.spatial_map is spat and worker.maintenance_interval == 0.5
    assert isinstance(worker.stop_event, threading.Event)


def test_poll_queue_increments_and_returns() -> None:
    """`poll_queue` fetches a batch and updates counters."""

    class DummyScheduler:
        def __init__(self) -> None:
            self.batch = [("episodic", None)]

        def next_batch(self, size: int) -> list[tuple[str, object]]:
            assert size == 1
            return self.batch

    worker = ConsolidationWorker(DummyScheduler(), torch.nn.Linear(1, 1), batch_size=1)
    batch = worker.poll_queue()
    assert batch == [("episodic", None)]
    assert worker.log_status()["batches"] == 1


def test_step_adapters_dispatches() -> None:
    """`step_adapters` routes items to the correct helper."""

    class DummyScheduler:
        def next_batch(self, size: int) -> list[tuple[str, object]]:  # pragma: no cover
            return []

    worker = ConsolidationWorker(DummyScheduler(), torch.nn.Linear(1, 1))
    calls: list[str] = []
    worker._step_episodic = lambda: calls.append("e")  # type: ignore[assignment]
    worker._step_semantic = lambda: calls.append("s")  # type: ignore[assignment]
    worker._step_fresh = lambda: calls.append("f")  # type: ignore[assignment]

    worker.step_adapters([("episodic", None), ("semantic", None), ("fresh", None)])
    assert calls == ["e", "s", "f"]


def test_step_semantic_uses_kg_features(caplog) -> None:
    """Semantic step uses KG embeddings and reports loss."""

    hidden = 2

    class DummyKG:
        def __init__(self) -> None:
            self.node_embeddings = {"n": np.ones(hidden, dtype=np.float32)}

        def retrieve(self, _q, k=1, radius=1):  # pragma: no cover - simple
            g = nx.MultiDiGraph()
            g.add_node("n")
            return g

        dim = hidden

    store = EpisodicStore(hidden)
    kg = DummyKG()
    scheduler = ReplayScheduler(
        store, kg, batch_mix=_BatchMix(episodic=0.0, semantic=1.0, fresh=0.0)
    )
    model = torch.nn.Linear(hidden, hidden)
    adapter = RelationalMemoryAdapter()
    worker = ConsolidationWorker(
        scheduler,
        model,
        relational_adapter=adapter,
        kg_store=kg,
        batch_size=1,
    )
    worker.rel_adapter = adapter
    losses: list[float] = []
    worker._optim_step = lambda loss: losses.append(float(loss))  # type: ignore[assignment]
    caplog.set_level(logging.DEBUG)
    worker._step_semantic()
    assert losses and losses[0] != 0
    assert "relational adapter step" in caplog.text


def test_handle_errors_logs_and_stops(caplog) -> None:
    """`handle_errors` logs exceptions and sets the stop flag."""

    class DummyScheduler:
        def next_batch(self, size: int) -> list[tuple[str, object]]:  # pragma: no cover
            return []

    worker = ConsolidationWorker(DummyScheduler(), torch.nn.Linear(1, 1))
    with caplog.at_level(logging.ERROR):
        result = worker.handle_errors(lambda: 42)
    assert result == 42 and not worker.stop_event.is_set()

    def boom() -> None:
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        assert worker.handle_errors(boom) is None
    assert worker.stop_event.is_set()
    assert "consolidation step failed" in caplog.text
