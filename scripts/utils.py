import logging
from typing import Optional

from hippo_mem.consolidation.worker import ConsolidationWorker
from hippo_mem.episodic.replay import ReplayScheduler
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.extract import extract_tuples
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.map import PlaceGraph

log = logging.getLogger(__name__)


def log_memory_status(
    store: EpisodicStore,
    kg: KnowledgeGraph,
    spatial_map: PlaceGraph,
    scheduler: Optional[ReplayScheduler],
    worker: Optional[ConsolidationWorker],
) -> None:
    """Emit status information for all memory components."""

    log.info(
        "episodic=%s kg=%s spatial=%s scheduler=%s worker=%s",
        store.log_status(),
        kg.log_status(),
        spatial_map.log_status(),
        scheduler.log_status() if scheduler else "disabled",
        worker.log_status() if worker else "disabled",
    )


def ingest_text(
    text: str,
    kg: KnowledgeGraph,
    *,
    threshold: float = 0.5,
    fasttrack: bool = True,
) -> int:
    """Extract tuples from ``text`` and insert them into ``kg``.

    Parameters
    ----------
    text : str
        Source document.
    kg : KnowledgeGraph
        Graph receiving extracted tuples.
    threshold : float, optional
        Confidence threshold passed to :func:`extract_tuples`.
    fasttrack : bool, optional
        When ``True`` use :meth:`KnowledgeGraph.ingest` which routes
        through the schema fast-track. Otherwise call
        :meth:`KnowledgeGraph.upsert` directly.

    Returns
    -------
    int
        Number of tuples inserted into ``kg``.
    """

    count = 0
    for tup in extract_tuples(text, threshold=threshold):
        if fasttrack:
            inserted = kg.ingest(tup)
        else:
            head, relation, tail, context, time, conf, prov = tup
            kg.upsert(head, relation, tail, context, time, conf, prov)
            inserted = True
        if inserted:
            count += 1
    return count
