from __future__ import annotations

import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hippo_mem.retrieval.faiss_index import FaissIndex


@settings(max_examples=25, deadline=None)
@given(
    vectors=st.lists(
        hnp.arrays(np.float32, 4, elements=st.floats(-1.0, 1.0)),
        min_size=1,
        max_size=10,
    ),
    data=st.data(),
)
def test_vector_index_roundtrip(vectors: list[np.ndarray], data: st.DataObject) -> None:
    """Round-trip add → update → remove maintains search rankings."""

    index = FaissIndex(dim=4)
    expected = []
    for vec in vectors:
        index.add(vec.tolist())
        expected.append(vec)

    query = data.draw(hnp.arrays(np.float32, 4, elements=st.floats(-1.0, 1.0)))

    def expected_ranking() -> list[int]:
        if not expected:
            return []
        mat = np.stack(expected)
        dists = np.linalg.norm(mat - query, axis=1)
        return np.argsort(dists, kind="stable").tolist()

    assert index.search(query.tolist(), k=len(expected)) == expected_ranking()

    update_indices = sorted(
        data.draw(
            st.sets(
                st.integers(min_value=0, max_value=len(expected) - 1),
            )
        ),
        reverse=True,
    )
    for idx in update_indices:
        new_vec = data.draw(hnp.arrays(np.float32, 4, elements=st.floats(-1.0, 1.0)))
        index.remove(idx)
        index.add(new_vec.tolist())
        expected.pop(idx)
        expected.append(new_vec)

    assert index.search(query.tolist(), k=len(expected)) == expected_ranking()

    remove_indices = sorted(
        data.draw(
            st.sets(
                st.integers(min_value=0, max_value=len(expected) - 1),
            )
        ),
        reverse=True,
    )
    for idx in remove_indices:
        index.remove(idx)
        expected.pop(idx)

    k = len(expected)
    results = index.search(query.tolist(), k=k if k else 1)
    assert results == (expected_ranking()[:k] if k else [])
