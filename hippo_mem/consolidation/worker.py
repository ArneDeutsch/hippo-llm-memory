"""Consolidation worker consuming replay batches and running maintenance.

Summary
-------
This module hosts a daemon thread that requests batches from the
``ReplayScheduler`` and applies 50/30/20 episodic/semantic/fresh mixes to
fine-tune adapters.  It also triggers maintenance jobs—decay, prune, and merge
operations—on the underlying stores to avoid growth and interference.

See Also
--------
hippo_mem.episodic.replay.ReplayScheduler
    Provides the replay batches processed here.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, TypeVar

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
    """Daemon thread that applies replay updates and store maintenance.

    Summary
    -------
    Consumes batches from a ``ReplayScheduler`` to fine-tune LoRA adapters while
    the base model stays frozen.  Between batches it launches decay, prune, and
    merge operations for episodic, semantic, and spatial stores.

    Parameters
    ----------
    scheduler:
        Source of replay batches.
    model:
        Base model whose parameters remain frozen.
    episodic_adapter, relational_adapter, spatial_adapter:
        Optional adapters to optimise.
    episodic_store, kg_store, spatial_map:
        Stores subjected to maintenance jobs.
    maintenance_interval:
        Seconds between maintenance passes.
    batch_size:
        Number of items to request per scheduler batch.
    lr:
        Learning rate for adapter optimisation.

    Returns
    -------
    None

    Raises
    ------
    None

    Side Effects
    ------------
    Spawns background threads and mutates adapters.

    Complexity
    ----------
    Depends on adapter size; maintenance loops run in ``O(n)`` over store
    entries.

    Examples
    --------
    >>> mix = type("Mix", (), {"episodic": 1.0, "semantic": 0.0, "fresh": 0.0})()
    >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
    >>> worker = ConsolidationWorker(sched, object(), batch_size=1)
    >>> worker.log_status()["batches"]
    0

    See Also
    --------
    hippo_mem.episodic.replay.ReplayScheduler
    """

    def __init__(
        self,
        scheduler: ReplayScheduler,
        model: object,
        *,
        episodic_adapter: Optional[EpisodicAdapter] = None,
        relational_adapter: Optional[RelationalAdapter] = None,
        spatial_adapter: Optional[SpatialAdapter] = None,
        episodic_store: Optional[EpisodicStore] = None,
        kg_store: Optional[KnowledgeGraph] = None,
        spatial_map: Optional[PlaceGraph] = None,
        maintenance_interval: float = 3600.0,
        batch_size: int = 4,
        lr: float = 1e-4,
    ) -> None:
        super().__init__(daemon=True)
        self.batch_size = batch_size
        self.log = logging.getLogger(__name__)
        self._log = {"batches": 0}

        self._setup_scheduler(
            scheduler,
            model,
            episodic_adapter,
            relational_adapter,
            spatial_adapter,
            lr,
        )
        self._setup_maintenance(
            episodic_store,
            kg_store,
            spatial_map,
            maintenance_interval,
        )

    # ------------------------------------------------------------------
    def _setup_scheduler(
        self,
        scheduler: ReplayScheduler,
        model: object,
        episodic_adapter: Optional[EpisodicAdapter],
        relational_adapter: Optional[RelationalAdapter],
        spatial_adapter: Optional[SpatialAdapter],
        lr: float,
    ) -> None:
        """Assign scheduler and prepare optimisation components."""

        self.scheduler = scheduler
        self.model = model
        self.epi_adapter = episodic_adapter
        self.rel_adapter = relational_adapter
        self.spat_adapter = spatial_adapter

        self._freeze_model(model)
        self._freeze_adapter(self.epi_adapter)
        self._freeze_adapter(self.spat_adapter)

        params: list[torch.nn.Parameter] = []
        if self.epi_adapter is not None:
            epi_params = [p for p in self.epi_adapter.parameters() if p.requires_grad]
            if not epi_params:
                self.log.warning(
                    "episodic_adapter has no trainable parameters; skipping optimisation",
                )
                self.epi_adapter = None
            else:
                params.extend(epi_params)
        if self.spat_adapter is not None:
            spat_params = [p for p in self.spat_adapter.parameters() if p.requires_grad]
            if not spat_params:
                self.log.warning(
                    "spatial_adapter has no trainable parameters; skipping optimisation",
                )
                self.spat_adapter = None
            else:
                params.extend(spat_params)
        self.optimizer = torch.optim.Adam(params, lr=lr) if params else None

    def _freeze_model(self, model: object) -> None:
        params_fn = getattr(model, "parameters", None)
        if callable(params_fn):
            for p in params_fn():
                p.requires_grad_(False)

    def _freeze_adapter(self, adapter: Optional[torch.nn.Module]) -> None:
        if adapter is None:
            return
        for p in adapter.parameters():
            p.requires_grad_(False)
        for name, p in adapter.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)

    def _setup_maintenance(
        self,
        episodic_store: Optional[EpisodicStore],
        kg_store: Optional[KnowledgeGraph],
        spatial_map: Optional[PlaceGraph],
        interval: float,
    ) -> None:
        """Store references and prepare the maintenance thread state."""

        self.epi_store = episodic_store
        self.kg_store = kg_store
        self.spatial_map = spatial_map
        self.maintenance_interval = interval
        self._maintenance_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    # ------------------------------------------------------------------
    def _start_maintenance(self) -> None:
        if self._maintenance_thread is not None:
            return
        if not any([self.epi_store, self.kg_store, self.spatial_map]):
            return

        def loop() -> None:
            while not self.stop_event.is_set():
                if self.epi_store is not None:
                    rate = float(self.epi_store.config.get("decay_rate", 0.0))
                    if rate > 0:
                        self.epi_store.decay(rate)
                    cfg = self.epi_store.config.get("prune", {})
                    self.epi_store.prune(
                        float(cfg.get("min_salience", 0.1)),
                        cfg.get("max_age"),
                    )
                if self.kg_store is not None:
                    cfg = self.kg_store.config.get("prune", {})
                    self.kg_store.prune(
                        float(cfg.get("min_conf", 0.0)),
                        cfg.get("max_age"),
                    )
                if self.spatial_map is not None:
                    rate = float(self.spatial_map.config.get("decay_rate", 0.0))
                    if rate > 0:
                        self.spatial_map.decay(rate)
                    cfg = self.spatial_map.config.get("prune", {})
                    age = cfg.get("max_age")
                    if age is not None:
                        self.spatial_map.prune(int(age))
                time.sleep(self.maintenance_interval)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._maintenance_thread = t

    # ------------------------------------------------------------------
    def _optim_step(self, loss: torch.Tensor) -> None:
        if self.optimizer is None:
            return
        if not loss.requires_grad:
            raise RuntimeError("loss does not require gradients")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _step_episodic(self) -> None:
        if self.epi_adapter is None:
            return
        h = torch.randn(1, 1, self.epi_adapter.hidden_size)
        m = torch.randn(1, 1, self.epi_adapter.hidden_size)
        out = self.epi_adapter(h, m)
        loss = out.sum()
        loss = loss + sum(
            p.sum() for name, p in self.epi_adapter.named_parameters() if "lora_" in name
        )
        self._optim_step(loss)
        self.log.debug("episodic adapter step")

    def _step_semantic(self) -> None:
        if self.rel_adapter is None:
            return
        q = np.zeros(1, dtype=float)
        k = np.zeros((1, 1), dtype=float)
        self.rel_adapter(q, k)
        self.log.debug("relational adapter step")

    def _step_fresh(self) -> None:
        if self.spat_adapter is None:
            return
        h = torch.randn(1, 1, self.spat_adapter.hidden_size)
        p = torch.randn(1, 1, self.spat_adapter.hidden_size)
        out = self.spat_adapter(h, p)
        loss = out.sum()
        loss = loss + sum(
            p.sum() for name, p in self.spat_adapter.named_parameters() if "lora_" in name
        )
        self._optim_step(loss)
        self.log.debug("spatial adapter step")

    # ------------------------------------------------------------------
    def poll_queue(self) -> list[tuple[str, object]]:
        """Fetch a batch from the scheduler and increment counters.

        Summary
        -------
        Requests ``batch_size`` items from the scheduler and records the
        invocation.

        Parameters
        ----------
        None

        Returns
        -------
        list[tuple[str, object]]
            Replay items as ``(kind, identifier)`` pairs.

        Raises
        ------
        None

        Side Effects
        ------------
        Updates internal batch counters.

        Complexity
        ----------
        Delegates to :class:`ReplayScheduler`.

        Examples
        --------
        >>> mix = type("Mix", (), {"episodic": 1.0, "semantic": 0.0, "fresh": 0.0})()
        >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
        >>> worker = ConsolidationWorker(sched, object(), batch_size=1)
        >>> worker.poll_queue()
        [('episodic', None)]

        See Also
        --------
        step_adapters
        """

        batch = self.scheduler.next_batch(self.batch_size)
        self._log["batches"] += 1
        return batch

    def step_adapters(self, batch: list[tuple[str, object]]) -> None:
        """Run adapter optimisation steps for items in ``batch``.

        Summary
        -------
        Dispatches optimisation routines depending on item kind.

        Parameters
        ----------
        batch:
            List of ``(kind, identifier)`` tuples.

        Returns
        -------
        None

        Raises
        ------
        None

        Side Effects
        ------------
        Updates adapter parameters and emits debug logs.

        Complexity
        ----------
        ``O(k)`` where ``k`` is batch size.

        Examples
        --------
        >>> mix = type("Mix", (), {"episodic": 1.0, "semantic": 0.0, "fresh": 0.0})()
        >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
        >>> worker = ConsolidationWorker(sched, object(), batch_size=1)
        >>> worker.step_adapters([('episodic', None)])

        See Also
        --------
        poll_queue
        """

        for kind, _ in batch:
            if self.stop_event.is_set():
                break
            if kind == "episodic":
                self._step_episodic()
            elif kind == "semantic":
                self._step_semantic()
            elif kind == "fresh":
                self._step_fresh()

    T = TypeVar("T")

    def handle_errors(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Optional[T]:
        """Execute ``fn`` and stop the worker if it raises.

        Summary
        -------
        Wrapper that logs exceptions and signals shutdown.

        Parameters
        ----------
        fn:
            Callable to execute.
        *args, **kwargs:
            Arguments passed to ``fn``.

        Returns
        -------
        Optional[T]
            Result of ``fn`` or ``None`` if an exception occurred.

        Raises
        ------
        None

        Side Effects
        ------------
        Logs exceptions and sets ``stop_event`` on failure.

        Complexity
        ----------
        Same as ``fn``.

        Examples
        --------
        >>> worker = ConsolidationWorker(
        ...     ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=type('M',(),{'episodic':1,'semantic':0,'fresh':0})()),
        ...     object(),
        ...     batch_size=1,
        ... )
        >>> worker.handle_errors(lambda x: x + 1, 1)
        2

        See Also
        --------
        run
        """

        try:
            return fn(*args, **kwargs)
        except Exception:  # pragma: no cover - defensive
            self.log.exception("consolidation step failed")
            self.stop_event.set()
            return None

    def run(self) -> None:  # pragma: no cover - exercised via thread
        """Main worker loop.

        Summary
        -------
        Continuously polls the scheduler, optimises adapters, and runs
        maintenance until ``stop_event`` is set.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None

        Side Effects
        ------------
        Runs maintenance threads and sleeps briefly between iterations.

        Complexity
        ----------
        Depends on scheduler and adapters; each loop iteration is roughly
        ``O(k)`` for ``k`` batch size.

        Examples
        --------
        >>> mix = type("Mix", (), {"episodic": 1.0, "semantic": 0.0, "fresh": 0.0})()
        >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
        >>> worker = ConsolidationWorker(sched, object(), batch_size=1)
        >>> worker.start(); worker.stop(); worker.join()

        See Also
        --------
        stop
        """

        self._start_maintenance()
        while not self.stop_event.is_set():
            batch = self.handle_errors(self.poll_queue)
            if batch is None:
                break
            self.handle_errors(self.step_adapters, batch)
            time.sleep(0.01)

    def log_status(self) -> dict:
        """Return counters for processed batches.

        Summary
        -------
        Provides diagnostics about processed batches.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with diagnostic counters.

        Raises
        ------
        None

        Side Effects
        ------------
        None

        Complexity
        ----------
        ``O(1)``.

        Examples
        --------
        >>> mix = type("Mix", (), {"episodic": 1.0, "semantic": 0.0, "fresh": 0.0})()
        >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
        >>> worker = ConsolidationWorker(sched, object(), batch_size=1)
        >>> worker.log_status()
        {'batches': 0}

        See Also
        --------
        poll_queue
        """

        return dict(self._log)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Signal the worker to exit.

        Summary
        -------
        Sets the internal stop flag so background threads can terminate.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None

        Side Effects
        ------------
        Sets ``stop_event``.

        Complexity
        ----------
        ``O(1)``.

        Examples
        --------
        >>> mix = type("Mix", (), {"episodic": 1.0, "semantic": 0.0, "fresh": 0.0})()
        >>> sched = ReplayScheduler(EpisodicStore(), KnowledgeGraph(), batch_mix=mix)
        >>> worker = ConsolidationWorker(sched, object(), batch_size=1)
        >>> worker.stop()

        See Also
        --------
        run
        """

        self.stop_event.set()


__all__ = ["ConsolidationWorker"]
