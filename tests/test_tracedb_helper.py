import numpy as np

from hippo_mem.episodic.db import TraceDB


def test_tracedb_exec_helper(tmp_path):
    path = tmp_path / "traces.sqlite"
    db = TraceDB(str(path))

    key = np.array([1.0, 2.0], dtype=np.float32)
    idx = db.insert(key, '{"a": 1}', ts=0.0, salience=0.1)

    # fetch via helper
    count = db._exec("SELECT COUNT(*) FROM traces", fetch="one")[0]
    assert count == 1

    db.update_value(idx, '{"a": 2}')
    db.update_key(idx, key + 1)

    value, stored_key, *_ = db.get(idx)
    assert value == '{"a": 2}'
    assert np.allclose(stored_key, key + 1)

    db.delete(idx)
    assert db.get(idx) is None
