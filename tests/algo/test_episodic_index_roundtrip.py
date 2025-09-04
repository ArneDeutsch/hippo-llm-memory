import numpy as np

from hippo_mem._faiss import faiss
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.episodic.types import TraceValue


def _reload_store(path: str, dim: int, ids: list[int]) -> EpisodicStore:
    """Recreate an :class:`EpisodicStore` loading keys for ``ids``."""
    store = EpisodicStore(dim=dim, db_path=path)
    for idx in ids:
        record = store.persistence.get(idx)
        if record is None:
            continue
        _, key_vec, *_ = record
        key_arr = key_vec.reshape(1, -1)
        faiss.normalize_L2(key_arr)
        store.index.add(key_arr, idx)
    return store


def test_kwta_collision_recall() -> None:
    """Colliding k-WTA keys are both retrievable."""
    store = EpisodicStore(dim=4, k_wta=1)
    a = np.array([1.0, 0.2, 0.0, 0.0], dtype="float32")
    b = np.array([0.9, 0.3, 0.0, 0.0], dtype="float32")
    id_a = store.write(a, TraceValue(provenance="a"))
    id_b = store.write(b, TraceValue(provenance="b"))
    res = store.recall(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), k=2)
    assert {r.id for r in res} == {id_a, id_b}


def test_add_delete_query_property() -> None:
    """add → query → delete → query yields empty result."""
    store = EpisodicStore(dim=4, k_wta=1)
    vec = np.array([0.5, 0.1, 0.0, 0.0], dtype="float32")
    idx = store.write(vec, TraceValue(provenance="x"))
    assert store.recall(vec, k=1)
    store.delete(idx)
    assert store.recall(vec, k=1) == []


def test_persistence_roundtrip(tmp_path) -> None:
    """Stored traces survive reload and deletions persist."""
    path = tmp_path / "traces.sqlite"
    vec = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
    store = EpisodicStore(dim=4, db_path=str(path))
    tid = store.write(vec, TraceValue(provenance="persist"))

    store = _reload_store(str(path), dim=4, ids=[tid])
    res = store.recall(vec, k=1)
    assert res and res[0].id == tid

    store.delete(tid)
    store = _reload_store(str(path), dim=4, ids=[tid])
    assert store.recall(vec, k=1) == []


def test_collision_rollback() -> None:
    """Colliding keys pruned then restored by rollback."""
    store = EpisodicStore(dim=4, k_wta=1)
    a = np.array([1.0, 0.2, 0.0, 0.0], dtype="float32")
    b = np.array([0.9, 0.3, 0.0, 0.0], dtype="float32")
    id_a = store.write(a, TraceValue(provenance="a"))
    id_b = store.write(b, TraceValue(provenance="b"))
    store.prune(min_salience=1.1)
    assert store.recall(a, k=2) == []
    store.rollback()
    res = store.recall(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), k=2)
    assert {r.id for r in res} == {id_a, id_b}
