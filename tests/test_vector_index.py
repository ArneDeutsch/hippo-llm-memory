import numpy as np
import pytest

from hippo_mem._faiss import faiss
from hippo_mem.episodic.index import VectorIndex


def _vec(values: list[float]) -> np.ndarray:
    mat = np.array([values], dtype="float32")
    faiss.normalize_L2(mat)
    return mat


def test_pending_and_search_training_flow() -> None:
    """Pending keys accumulate and search works before and after training."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=2)
    v1 = _vec([1.0, 0.0, 0.0])
    index.add(v1, 1)
    assert not index.is_trained
    assert index.ntotal == 0
    assert len(index._pending_keys) == 1
    dist, ids = index.search(v1, k=1)
    assert ids[0][0] == -1
    v2 = _vec([0.0, 1.0, 0.0])
    index.add(v2, 2)
    assert index.is_trained
    assert index.ntotal == 2
    assert not index._pending_keys
    dist, ids = index.search(v1, k=1)
    assert ids[0][0] == 1


def test_update_mutates_index() -> None:
    """Updating a vector replaces the previous id's value."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    v1 = _vec([1.0, 0.0, 0.0])
    index.add(v1, 1)
    new_v1 = _vec([0.0, 1.0, 0.0])
    index.update(new_v1, 1)
    dist, ids = index.search(new_v1, k=1)
    assert ids[0][0] == 1


def test_remove_mutates_index() -> None:
    """Removing an id shrinks the index and excludes the vector."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    v1 = _vec([1.0, 0.0, 0.0])
    v2 = _vec([0.0, 1.0, 0.0])
    index.add(v1, 1)
    index.add(v2, 2)
    index.remove(1)
    assert index.ntotal == 1
    dist, ids = index.search(v1, k=1)
    assert ids[0][0] != 1


def test_duplicate_ids_are_supported() -> None:
    """Adding vectors with duplicate ids returns both on search."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    v1 = _vec([1.0, 0.0, 0.0])
    v2 = _vec([0.0, 1.0, 0.0])
    index.add(v1, 42)
    index.add(v2, 42)
    assert index.ntotal == 2
    dist, ids = index.search(v2, k=2)
    assert list(ids[0]) == [42, 42]


def test_pq_train_requires_data() -> None:
    """Training a PQ index on empty data raises an error."""
    index = VectorIndex(dim=4, index_str="PQ2", train_threshold=1)
    empty = np.empty((0, 4), dtype="float32")
    with pytest.raises(RuntimeError):
        index.index.train(empty)


def test_add_raises_on_dimension_mismatch() -> None:
    """Adding a vector with wrong dims fails."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    wrong = _vec([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(Exception):
        index.add(wrong, 1)


def test_search_raises_on_dimension_mismatch() -> None:
    """Searching with wrong dims fails."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    v1 = _vec([1.0, 0.0, 0.0])
    index.add(v1, 1)
    wrong = _vec([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(Exception):
        index.search(wrong, k=1)


def test_remove_rejects_invalid_ids() -> None:
    """Removing negative or missing ids leaves the index intact."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    v1 = _vec([1.0, 0.0, 0.0])
    v2 = _vec([0.0, 1.0, 0.0])
    index.add(v1, 1)
    index.add(v2, 2)
    assert index.ntotal == 2
    index.remove(-1)
    index.remove(99)
    assert index.ntotal == 2
