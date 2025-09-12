# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import numpy as np

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.episodic.types import TraceValue


def test_recall_filters_by_context_key() -> None:
    store = EpisodicStore(2)
    store.write(
        np.array([1.0, 0.0], dtype=np.float32),
        TraceValue(provenance="a"),
        context_key="A",
    )
    store.write(
        np.array([0.0, 1.0], dtype=np.float32),
        TraceValue(provenance="b"),
        context_key="B",
    )
    res = store.recall(np.array([1.0, 0.0], dtype=np.float32), 2, context_key="A")
    assert len(res) == 1
    assert res[0].value.provenance == "a"
