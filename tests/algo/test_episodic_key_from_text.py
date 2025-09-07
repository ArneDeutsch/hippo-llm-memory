import numpy as np

from hippo_eval.eval.harness import _episodic_key_from_text


def test_key_from_text_normalises_and_sparsifies() -> None:
    dense, sparse = _episodic_key_from_text("hello", dim=4, k=2)
    assert np.isclose(np.linalg.norm(dense), 1.0)
    assert len(sparse.indices) == 2
    assert len(sparse.values) == 2
    assert sparse.dim == 4
