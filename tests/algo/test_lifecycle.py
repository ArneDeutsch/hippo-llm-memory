import time
import types

import pytest

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.map import PlaceGraph


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (EpisodicStore, {"dim": 2}),
        (KnowledgeGraph, {}),
        (PlaceGraph, {}),
    ],
)
def test_double_start_background_tasks(factory, kwargs):
    """Starting background tasks twice uses the same thread and stops cleanly."""

    store = factory(**kwargs)
    calls = {"n": 0}

    def tick(self, event):
        calls["n"] += 1

    store._maintenance_tick = types.MethodType(tick, store)  # type: ignore[assignment]
    store.start_background_tasks(interval=0.01)
    thread1 = store._task_manager._thread  # type: ignore[attr-defined]
    store.start_background_tasks(interval=0.01)
    thread2 = store._task_manager._thread  # type: ignore[attr-defined]
    assert thread1 is thread2
    time.sleep(0.025)
    store.stop_background_tasks()
    stopped = calls["n"]
    time.sleep(0.02)
    assert calls["n"] == stopped
    assert store._task_manager._thread is None  # type: ignore[attr-defined]
    assert calls["n"] >= 1
