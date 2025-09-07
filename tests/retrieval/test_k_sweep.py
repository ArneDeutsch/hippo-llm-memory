import numpy as np

from hippo_mem.episodic.store import EpisodicStore


def test_k_sweep_hit_rates_monotonic():
    dim = 8
    store = EpisodicStore(dim, k_wta=2)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, dim), dtype="float32")
    target_id = store.write(vectors[0], "t0")
    for vec in vectors[1:]:
        store.write(vec, "t")
    query = vectors[0]
    ks = [1, 4, 8, 16]
    hit_rates = store.hit_rate_sweep(query, target_id, ks)
    assert hit_rates[0] <= hit_rates[1] <= hit_rates[2] <= hit_rates[3]
    assert any(hr > 0 for hr in hit_rates)
