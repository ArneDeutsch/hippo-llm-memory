import logging
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.map import PlaceGraph
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.consolidation.worker import ConsolidationWorker

log = logging.getLogger(__name__)

def log_memory_status(
    store: EpisodicStore,
    kg: KnowledgeGraph,
    spatial_map: PlaceGraph,
    scheduler: ReplayScheduler,
    worker: ConsolidationWorker,
) -> None:
    """Emit status information for all memory components."""

    log.info(
        "episodic=%s kg=%s spatial=%s scheduler=%s worker=%s",
        store.log_status(),
        kg.log_status(),
        spatial_map.log_status(),
        scheduler.log_status(),
        worker.log_status(),
    )

