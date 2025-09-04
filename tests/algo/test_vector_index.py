import numpy as np
import pytest

from hippo_mem._faiss import faiss
from hippo_mem.episodic.index import NumpyIndex, VectorIndex


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
    assert not index.is_trained
    assert index.ntotal == 0
    with pytest.raises(RuntimeError, match="Number of training points"):
        index.index.train(empty)
    assert not index.is_trained
    assert index.ntotal == 0


def test_add_raises_on_dimension_mismatch() -> None:
    """Adding a vector with wrong dims fails."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    wrong = _vec([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(AssertionError):
        index.add(wrong, 1)


def test_search_raises_on_dimension_mismatch() -> None:
    """Searching with wrong dims fails."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    v1 = _vec([1.0, 0.0, 0.0])
    index.add(v1, 1)
    wrong = _vec([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(AssertionError):
        index.search(wrong, k=1)


def test_search_empty_index_returns_placeholder() -> None:
    """Searching an empty index returns placeholder ids."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    query = _vec([1.0, 0.0, 0.0])
    dist, ids = index.search(query, k=1)
    assert ids[0][0] == -1


def test_remove_rejects_invalid_ids() -> None:
    """Removing negative, overflow, or missing ids leaves the index intact."""
    index = VectorIndex(dim=3, index_str="IVF1,Flat", train_threshold=1)
    v1 = _vec([1.0, 0.0, 0.0])
    v2 = _vec([0.0, 1.0, 0.0])
    index.add(v1, 1)
    index.add(v2, 2)
    assert index.ntotal == 2
    index.remove(-1)
    index.remove(99)
    index.remove(2**63 - 1)
    assert index.ntotal == 2


def test_pq_search_untrained_raises() -> None:
    """Searching a PQ index before training raises an error."""
    index = VectorIndex(dim=4, index_str="PQ2", train_threshold=2)
    v1 = _vec([1.0, 0.0, 0.0, 0.0])
    index.add(v1, 1)
    assert not index.is_trained
    with pytest.raises(RuntimeError):
        index.search(v1, k=1)


def test_numpy_index_collision_and_deletion() -> None:
    """NumpyIndex overwrites ids and ignores missing removals."""

    index = NumpyIndex(dim=3)
    v1 = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    v2 = np.array([[0.0, 1.0, 0.0]], dtype="float32")
    index.add(v1, 7)
    index.add(v2, 7)  # collision
    assert index.ntotal == 1
    dist, ids = index.search(v2, k=1)
    assert ids[0][0] == 7
    index.remove(99)  # out-of-range
    assert index.ntotal == 1
    index.remove(7)
    assert index.ntotal == 0
    dist, ids = index.search(v1, k=1)
    assert ids[0][0] == -1
