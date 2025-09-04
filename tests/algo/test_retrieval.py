import time

import numpy as np
import pytest
import torch
from torch import nn

from hippo_mem.common.retrieval import build_meta, retrieve_and_pack_base
from hippo_mem.retrieval.embed import embed_text
from hippo_mem.retrieval.faiss_index import FaissIndex


def test_embed_text_is_deterministic() -> None:
    """Embedding the same text twice yields identical vectors with padding."""
    vec1 = embed_text("abc", dim=5)
    vec2 = embed_text("abc", dim=5)
    assert vec1 == vec2
    assert len(vec1) == 5
    assert vec1[3:] == [0.0, 0.0]
    assert vec1[0] == pytest.approx(ord("a") / 255.0)


def test_embed_text_truncates_long_input() -> None:
    """Long strings are truncated to the target dimension."""
    text = "abcdef"
    vec = embed_text(text, dim=3)
    assert len(vec) == 3
    assert vec == [pytest.approx(ord(c) / 255.0) for c in "abc"]


def test_faiss_index_add_and_search() -> None:
    """Vectors can be added and queried by nearest neighbour."""
    index = FaissIndex(dim=4)
    index.add([1.0, 0.0, 0.0, 0.0])
    index.add([0.0, 1.0, 0.0, 0.0])
    assert index.search([0.9, 0.1, 0.0, 0.0], k=1) == [0]
    assert index.search([0.1, 0.9, 0.0, 0.0], k=1) == [1]


def test_faiss_index_train_smoke() -> None:
    """Training a PQ index is a no-op without FAISS but should not error."""
    index = FaissIndex(dim=8, use_pq=True)
    data = [[1.0 if i % 8 == j else 0.0 for j in range(8)] for i in range(256)]
    index.train(data)
    index.add(data[0])
    assert index.search(data[0], k=1) == [0]


def test_faiss_index_remove() -> None:
    """Vectors can be removed from the index and the size updates."""
    index = FaissIndex(dim=4, use_pq=True, m=2)
    train_data = [[1.0 if i % 4 == j else 0.0 for j in range(4)] for i in range(256)]
    index.train(train_data)
    for vec in train_data[:2]:
        index.add(vec)
    assert len(index) == 2
    assert index.search(train_data[1], k=1) == [1]
    index.remove(1)
    assert len(index) == 1
    assert index.search(train_data[1], k=1) != [1]


def test_faiss_index_edge_cases() -> None:
    """Search should handle empty and untrained PQ indices consistently."""
    index = FaissIndex(dim=4, use_pq=True, m=2)
    query = [1.0, 0.0, 0.0, 0.0]

    # Search before any training or vectors have been added.
    backend_index = getattr(index._backend, "index", None)
    if backend_index is not None:
        assert not backend_index.is_trained
    assert index.search(query, k=3) == []

    # Train the index but do not add vectors yet; still expect no hits.
    train_data = [[1.0 if i % 4 == j else 0.0 for j in range(4)] for i in range(256)]
    index.train(train_data)
    backend_index = getattr(index._backend, "index", None)
    if backend_index is not None:
        assert backend_index.is_trained
    assert index.search(query, k=2) == []

    # After adding vectors the nearest neighbour should be returned.
    for vec in train_data[:2]:
        index.add(vec)
    res = index.search(query, k=5)
    assert res[0] == 0 and len(res) == 2


def test_faiss_index_update_and_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Updating vectors and invalid operations raise appropriate errors."""
    index = FaissIndex(dim=2)
    # Initial add and update by remove+add cycle.
    index.add([1.0, 0.0])
    index.add([0.0, 1.0])
    index.remove(1)
    index.add([0.5, 0.5])  # updated vector
    assert len(index) == 2
    assert index.search([0.6, 0.4], k=1) == [1]

    # Dimension mismatches.
    with pytest.raises(ValueError):
        index.add([1.0])
    with pytest.raises(ValueError):
        index.search([0.0], k=1)

    # Negative index always raises.
    with pytest.raises(ValueError):
        index.remove(-1)

    # Patch to numpy fallback to exercise out-of-range error.
    monkeypatch.setattr("hippo_mem.retrieval.faiss_index.faiss", None)
    fallback = FaissIndex(dim=2)
    fallback.add([0.0, 1.0])
    with pytest.raises(IndexError):
        fallback.remove(10)


def test_build_meta_schema() -> None:
    """``build_meta`` emits the expected telemetry fields."""

    start = time.perf_counter()
    meta = build_meta("kind", start, hits=2, k=3, bsz=1)
    assert set(meta.keys()) == {"source", "k", "batch_size", "latency_ms", "hit_rate"}


def test_retrieve_and_pack_meta_schema() -> None:
    """Retrieval wrapper constructs metadata with the centralised schema."""

    def retrieve():
        yield np.ones((1, 2), dtype=np.float32), 1

    mem = retrieve_and_pack_base(
        retrieve,
        k=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        proj=nn.Identity(),
        build_meta_fn=lambda start, hits, k, bsz: build_meta("kind", start, hits, k, bsz=bsz),
        telemetry_key="episodic",
    )
    assert set(mem.meta.keys()) == {"source", "k", "batch_size", "latency_ms", "hit_rate"}
