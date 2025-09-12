# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Tests for episodic store content fields and key diversity."""

import numpy as np

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.episodic.types import TraceValue


def test_fields_and_knn_recall() -> None:
    """Store writes content fields and recalls from partial cues."""

    dim = 16
    store = EpisodicStore(dim=dim, key_noise=0.01, seed=0)
    rng = np.random.default_rng(0)
    ids: list[int] = []
    keys: list[np.ndarray] = []
    for i in range(20):
        vec = rng.normal(size=dim).astype("float32")
        keys.append(vec)
        value = TraceValue(
            tokens_span=(i, i + 1),
            entity_slots={"id": i},
            state_sketch=[i],
            salience_tags=["novel"],
        )
        ids.append(store.write(vec, value))

    rows = store.persistence.all()
    filled = sum(
        1
        for _idx, val, _key, _ts, _sal in rows
        if val.tokens_span and val.entity_slots and val.state_sketch and val.salience_tags
    )
    assert filled / len(rows) >= 0.8

    key_matrix = np.vstack([k for _id, _v, k, _ts, _s in rows])
    uniq = np.unique(key_matrix.round(decimals=6), axis=0).shape[0]
    assert uniq >= len(rows) / 2

    hits = 0
    for vec, tid in zip(keys, ids):
        cue = vec.copy()
        cue[: dim // 2] = 0
        rec = store.recall(cue, 5)
        if tid in [t.id for t in rec]:
            hits += 1
    assert hits / len(keys) >= 0.7
