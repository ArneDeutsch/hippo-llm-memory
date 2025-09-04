import numpy as np
import pytest

from hippo_mem._faiss import faiss
from hippo_mem.episodic.index import VectorIndex


@pytest.mark.parametrize("dim,pq_m", [(4, 2), (8, 4)])
def test_pq_index_reload_equivalence(tmp_path, dim: int, pq_m: int) -> None:
    """PQ index retains neighbor results after save and reload."""
    rng = np.random.default_rng(0)
    n = 32
    vecs = rng.standard_normal((n, dim)).astype("float32")
    faiss.normalize_L2(vecs)
    index = VectorIndex(dim=dim, index_str=f"PQ{pq_m}x4", train_threshold=n)
    for i, vec in enumerate(vecs):
        index.add(vec[np.newaxis, :], i)
    assert index.is_trained
    assert index.ntotal == n
    query = vecs[0:1]
    expected = int(np.argmax(vecs @ query.T))
    dist_before, ids_before = index.search(query, k=1)
    assert ids_before[0][0] == expected
    index_path = tmp_path / "faiss.idx"
    faiss.write_index(index.index, str(index_path))
    reloaded = faiss.read_index(str(index_path))
    dist_after, ids_after = reloaded.search(query, k=1)
    assert np.allclose(dist_before, dist_after)
    assert np.array_equal(ids_before, ids_after)
