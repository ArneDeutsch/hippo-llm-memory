import numpy as np

import hippo_mem.common.io as io
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.episodic.types import TraceValue


def test_dense_keys_written(tmp_path):
    store = EpisodicStore(4, db_path=str(tmp_path / "epis.db"), k_wta=2)
    rng = np.random.default_rng(0)
    for i in range(5):
        store.write(rng.random(4, dtype="float32"), TraceValue(provenance=str(i)))
    store.save(str(tmp_path), "sess", replay_samples=5, gate_attempts=5)
    rows = list(io.read_jsonl(tmp_path / "sess" / "episodic.jsonl"))
    assert len(rows) == 5
    for rec in rows:
        key = np.asarray(rec["key"], dtype="float32")
        assert key.shape == (4,)
        assert np.linalg.norm(key) > 0
