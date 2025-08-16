"""Background consolidation worker for replay-based finetuning.

The worker runs in a daemon thread and periodically asks the
:class:`~hippo_mem.episodic.replay.ReplayScheduler` for replay batches.  Based
on the kind of replay item it performs a tiny optimisation step on the
corresponding LoRA adapter while keeping the base model frozen.  This is a
light‑weight stand‑in for a more realistic consolidation process.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
import torch

from hippo_mem.episodic.adapter import EpisodicAdapter
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.adapter import RelationalAdapter
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.adapter import SpatialAdapter
from hippo_mem.spatial.map import PlaceGraph


class ConsolidationWorker(threading.Thread):
    """Background thread that fine‑tunes memory adapters using replay."""

    def __init__(
        self,
        scheduler: ReplayScheduler,
        model: object,
        *,
        episodic_adapter: Optional[EpisodicAdapter] = None,
        relational_adapter: Optional[RelationalAdapter] = None,
        spatial_adapter: Optional[SpatialAdapter] = None,
        episodic_store: Optional[EpisodicStore] = None,
        relational_store: Optional[KnowledgeGraph] = None,
        spatial_store: Optional[PlaceGraph] = None,
        batch_size: int = 4,
        lr: float = 1e-4,
        maintenance_interval: float = 600.0,
    ) -> None:
        super().__init__(daemon=True)
        self.scheduler = scheduler
        self.model = model
        self.epi_adapter = episodic_adapter
        self.rel_adapter = relational_adapter
        self.spat_adapter = spatial_adapter
        self.epi_store = episodic_store
        self.rel_store = relational_store
        self.spat_store = spatial_store
        self.batch_size = batch_size
        self.stop_event = threading.Event()
        self.log = logging.getLogger(__name__)
        self._log = {"batches": 0}
        self._maintenance_interval = maintenance_interval
        self._maint_thread: Optional[threading.Thread] = None

        # Freeze the base model parameters if possible.
        params_fn = getattr(model, "parameters", None)
        if callable(params_fn):
            for p in params_fn():
                p.requires_grad_(False)

        if episodic_adapter is not None:
            for p in episodic_adapter.parameters():
                p.requires_grad_(False)
            for name, p in episodic_adapter.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)
        if spatial_adapter is not None:
            for p in spatial_adapter.parameters():
                p.requires_grad_(False)
            for name, p in spatial_adapter.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)

        # Optimise only the parameters of the adapters.
        params = []
        if episodic_adapter is not None:
            params.extend(episodic_adapter.parameters())
        if spatial_adapter is not None:
            params.extend(spatial_adapter.parameters())
        # ``RelationalAdapter`` is stateless in this toy setup.
        self.optimizer = torch.optim.Adam(params, lr=lr) if params else None

        if any([self.epi_store, self.rel_store, self.spat_store]):
            self._maint_thread = threading.Thread(
                target=self._maintenance_loop, daemon=True
            )
            self._maint_thread.start()

    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - exercised via thread
        """Main worker loop."""

        while not self.stop_event.is_set():
            batch = self.scheduler.next_batch(self.batch_size)
            self._log["batches"] += 1
            for kind, _ in batch:
                if self.stop_event.is_set():
                    break
                if kind == "episodic" and self.epi_adapter is not None:
                    h = torch.randn(1, 1, self.epi_adapter.hidden_size)
                    m = torch.randn(1, 1, self.epi_adapter.hidden_size)
                    out = self.epi_adapter(h, m)
                    loss = out.sum()
                    loss = loss + sum(
                        p.sum()
                        for name, p in self.epi_adapter.named_parameters()
                        if "lora_" in name
                    )
                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    self.log.debug("episodic adapter step")
                elif kind == "semantic" and self.rel_adapter is not None:
                    q = np.zeros(1, dtype=float)
                    k = np.zeros((1, 1), dtype=float)
                    self.rel_adapter(q, k)
                    self.log.debug("relational adapter step")
                elif kind == "fresh" and self.spat_adapter is not None:
                    h = torch.randn(1, 1, self.spat_adapter.hidden_size)
                    p = torch.randn(1, 1, self.spat_adapter.hidden_size)
                    out = self.spat_adapter(h, p)
                    loss = out.sum()
                    loss = loss + sum(
                        p.sum()
                        for name, p in self.spat_adapter.named_parameters()
                        if "lora_" in name
                    )
                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                    self.optimizer.step()
                    self.log.debug("spatial adapter step")
            time.sleep(0.01)

    def log_status(self) -> dict:
        """Return counters for processed batches."""

        return dict(self._log)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Signal the worker to exit."""

        self.stop_event.set()
        if self._maint_thread is not None:
            self._maint_thread.join(timeout=0.1)

    # ------------------------------------------------------------------
    def _maintenance_loop(self) -> None:  # pragma: no cover - thread
        stores = [self.epi_store, self.rel_store, self.spat_store]
        while not self.stop_event.is_set():
            time.sleep(self._maintenance_interval)
            for store in stores:
                if store is None:
                    continue
                rate = float(getattr(store, "config", {}).get("decay_rate", 0.0))
                if rate > 0 and hasattr(store, "decay"):
                    store.decay(rate)
                cfg = getattr(store, "config", {}).get("prune", {})
                if cfg and hasattr(store, "prune"):
                    if isinstance(cfg, dict):
                        store.prune(**cfg)
                    else:
                        store.prune(cfg)
                if hasattr(store, "_log"):
                    store._log["maintenance"] = store._log.get("maintenance", 0) + 1


__all__ = ["ConsolidationWorker"]
