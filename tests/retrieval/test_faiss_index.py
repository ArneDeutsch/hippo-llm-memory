import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hippo_mem.retrieval.faiss_index import FaissIndex


@st.composite
def vector_sets(draw):
    dim = draw(st.integers(min_value=2, max_value=8))
    n = draw(st.integers(min_value=1, max_value=5))
    vectors = draw(hnp.arrays(dtype="float32", shape=(n, dim), elements=st.floats(-1, 1)))
    qid = draw(st.integers(min_value=0, max_value=n - 1))
    k = draw(st.integers(min_value=1, max_value=n))
    return vectors, qid, k


@given(vector_sets())
def test_round_trip_and_topk(data) -> None:
    vectors, qid, k = data
    index = FaissIndex(dim=vectors.shape[1])
    index.train(vectors)
    for vec in vectors:
        index.add(vec)
    query = vectors[qid]
    result = index.search(query, k=k)
    dists = np.linalg.norm(vectors - query, axis=1)
    threshold = np.partition(dists, k - 1)[k - 1]
    expected = {i for i, d in enumerate(dists) if d <= threshold}
    assert set(result).issubset(expected)
    returned = [dists[r] for r in result]
    assert returned == sorted(returned)


def test_dimension_mismatch() -> None:
    index = FaissIndex(dim=3)
    with pytest.raises(ValueError):
        index.add([0.0, 0.0])
    index.add([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        index.search([0.0, 0.0], k=1)


def test_search_before_train_returns_empty() -> None:
    index = FaissIndex(dim=4, use_pq=True, m=2)
    result = index.search([0.0, 0.0, 0.0, 0.0], k=1)
    assert result == []


def test_remove_missing_id_raises() -> None:
    index = FaissIndex(dim=2)
    index.add([0.0, 0.0])
    with pytest.raises(IndexError):
        index.remove(5)
