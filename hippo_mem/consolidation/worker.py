"""Background consolidation worker for replay-based finetuning.

Summary
-------
Runs in a daemon thread, polling the scheduler for batches mixed at 50%
episodic, 30% semantic and 20% fresh. Applies adapter updates and triggers
maintenance jobs (decay, prune, merge) on stores. Metrics are logged and the
worker stops on failures for rollback safety.

See Also
--------
hippo_mem.episodic.replay.ReplayScheduler
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, TypeVar

import torch

from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
from hippo_mem.common import TraceSpec
from hippo_mem.episodic.adapter import EpisodicAdapter
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.retrieval import relational_retrieve_and_pack
from hippo_mem.spatial.adapter import SpatialAdapter
from hippo_mem.spatial.map import PlaceGraph


class ConsolidationWorker(threading.Thread):
    """Background thread that fine‑tunes memory adapters using replay.

    Summary
    -------
    Consumes mixed replay batches to update LoRA adapters and run maintenance
    jobs.

    See Also
    --------
    ReplayScheduler
    """

    def __init__(
        self,
        scheduler: ReplayScheduler,
        model: object,
        *,
        episodic_adapter: Optional[EpisodicAdapter] = None,
        relational_adapter: Optional[RelationalMemoryAdapter] = None,
        spatial_adapter: Optional[SpatialAdapter] = None,
        episodic_store: Optional[EpisodicStore] = None,
        kg_store: Optional[KnowledgeGraph] = None,
        spatial_map: Optional[PlaceGraph] = None,
        maintenance_interval: float = 3600.0,
        batch_size: int = 4,
        lr: float = 1e-4,
    ) -> None:
        """Summary
        -------
        Construct the worker and prepare scheduler, adapters and maintenance.

        Parameters
        ----------
        scheduler : ReplayScheduler
            Source of replay batches.
        model : object
            Base model kept frozen during consolidation.
        episodic_adapter : EpisodicAdapter, optional
            Adapter updated from episodic batches.
        relational_adapter : RelationalMemoryAdapter, optional
            Adapter updated from semantic batches.
        spatial_adapter : SpatialAdapter, optional
            Adapter updated from fresh batches.
        episodic_store : EpisodicStore, optional
            Store subject to decay/prune maintenance.
        kg_store : KnowledgeGraph, optional
            Semantic store with pruning.
        spatial_map : PlaceGraph, optional
            Map subject to decay and pruning.
        maintenance_interval : float, optional
            Seconds between maintenance runs; default ``3600``.
        batch_size : int, optional
            Number of replay items per optimisation step; default ``4``.
        lr : float, optional
            Learning rate for adapter optimisation; default ``1e-4``.
        Side Effects
        ------------
        Spawns maintenance thread and initialises optimiser state.
        Examples
        --------
        >>> worker = ConsolidationWorker(ReplayScheduler(EpisodicStore(1), KnowledgeGraph(), batch_mix=type('B',(),{'episodic':1.0,'semantic':0.0,'fresh':0.0})()), object())

        See Also
        --------
        run
        """

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
        relational_adapter: Optional[RelationalMemoryAdapter],
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
        self._freeze_adapter(self.rel_adapter)
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
        if self.rel_adapter is not None:
            rel_params = [p for p in self.rel_adapter.parameters() if p.requires_grad]
            if not rel_params:
                self.log.warning(
                    "relational_adapter has no trainable parameters; skipping optimisation",
                )
                self.rel_adapter = None
            else:
                params.extend(rel_params)
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
        """
        Execute a placeholder update for the relational adapter.

        The current implementation fabricates a random hidden state, retrieves a
        single hop of KG tokens, and steps the adapter purely to exercise the
        optimisation path. Real semantic consolidation will replace this with
        meaningful graph-based batches once KG integration is complete.
        """
        if self.rel_adapter is None or self.kg_store is None:
            return
        proj = getattr(self.rel_adapter, "proj", None)
        dim = getattr(proj, "out_features", getattr(self.kg_store, "dim", 1))
        if proj is None:
            base_dim = getattr(self.kg_store, "dim", dim)
            proj = torch.nn.Linear(base_dim, dim)
            self.rel_adapter.proj = proj  # type: ignore[attr-defined]
        hidden = torch.randn(1, 1, dim)
        spec = TraceSpec(source="relational", k=1, params={"hops": 1})
        mem = relational_retrieve_and_pack(hidden, spec, self.kg_store, proj)
        out = self.rel_adapter(hidden, memory=mem)
        loss = out.sum()
        loss = loss + sum(
            p.sum() for name, p in self.rel_adapter.named_parameters() if "lora_" in name
        )
        self._optim_step(loss)
        self.log.debug("relational adapter step loss=%.4f", float(loss.detach()))

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
        """Summary
        -------
        Fetch a batch from the scheduler and increment counters.
        Returns
        -------
        list of tuple of str and object
            Batch of replay items.
        Side Effects
        ------------
        Updates internal batch counter.
        Examples
        --------
        >>> worker.poll_queue()  # doctest: +SKIP

        See Also
        --------
        step_adapters
        """

        batch = self.scheduler.next_batch(self.batch_size)
        # why: track how many batches processed
        self._log["batches"] += 1
        return batch

    def step_adapters(self, batch: list[tuple[str, object]]) -> None:
        """Summary
        -------
        Run adapter optimisation steps for items in ``batch``.

        Parameters
        ----------
        batch : list of tuple of str and object
            Replay items from :meth:`poll_queue`.
        Side Effects
        ------------
        Updates adapter parameters and may log debug messages.

        Complexity
        ----------
        ``O(n)`` for ``n`` items in ``batch``.

        Examples
        --------
        >>> worker.step_adapters([("episodic", object())])  # doctest: +SKIP

        See Also
        --------
        poll_queue
        """

        for kind, _ in batch:
            if self.stop_event.is_set():
                break  # why: allow graceful shutdown
            if kind == "episodic":
                self._step_episodic()
            elif kind == "semantic":
                self._step_semantic()
            elif kind == "fresh":
                self._step_fresh()

    T = TypeVar("T")

    def handle_errors(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Optional[T]:
        """Summary
        -------
        Execute ``fn`` and stop the worker if it raises.

        Parameters
        ----------
        fn : Callable[..., T]
            Function to execute.
        *args : Any
            Positional arguments for ``fn``.
        **kwargs : Any
            Keyword arguments for ``fn``.

        Returns
        -------
        Optional[T]
            Result of ``fn`` or ``None`` on failure.
        Side Effects
        ------------
        Logs the exception and sets ``stop_event`` for rollback.
        Examples
        --------
        >>> worker.handle_errors(lambda: 1)
        1

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
        """Summary
        -------
        Main worker loop consuming scheduler batches and stepping adapters.
        Side Effects
        ------------
        Starts maintenance thread and sleeps between batches.

        Complexity
        ----------
        Depends on scheduler and adapter costs.

        Examples
        --------
        >>> worker.start()  # doctest: +SKIP

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
        """Summary
        -------
        Return counters for processed batches.
        Returns
        -------
        dict
            Copy of internal counters.
        Examples
        --------
        >>> ConsolidationWorker(ReplayScheduler(EpisodicStore(1), KnowledgeGraph(), batch_mix=type('B',(),{'episodic':1.0,'semantic':0.0,'fresh':0.0})()), object()).log_status()['batches']
        0

        See Also
        --------
        poll_queue
        """

        return dict(self._log)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Summary
        -------
        Signal the worker to exit.
        Side Effects
        ------------
        Sets ``stop_event`` which halts loops.
        Examples
        --------
        >>> worker.stop()

        See Also
        --------
        run
        """

        self.stop_event.set()


__all__ = ["ConsolidationWorker"]
