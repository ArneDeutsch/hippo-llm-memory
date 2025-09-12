# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import numpy as np
import pytest

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


def test_tracedb_decay_fetch_restore(tmp_path):
    path = tmp_path / "traces.sqlite"
    db = TraceDB(str(path))

    keys = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
    ]
    values = ['{"a": 1}', '{"b": 2}', '{"c": 3}']
    saliences = [0.1, 0.2, 0.05]
    tss = [0.0, 10.0, 20.0]

    ids = [db.insert(k, v, ts=t, salience=s) for k, v, t, s in zip(keys, values, tss, saliences)]

    prev = db.decay(0.5)
    assert dict(prev) == {i: s for i, s in zip(ids, saliences)}
    for idx, s in zip(ids, saliences):
        _, _, _, new_s = db.get(idx)
        assert new_s == pytest.approx(s * 0.5)

    rows = db.fetch_prune_candidates(min_salience=0.06, cutoff=None)
    assert sorted(r[0] for r in rows) == [ids[0], ids[2]]

    rows_ts = db.fetch_prune_candidates(min_salience=None, cutoff=5.0)
    assert sorted(r[0] for r in rows_ts) == [ids[0]]

    rows_both = db.fetch_prune_candidates(min_salience=0.01, cutoff=5.0)
    assert sorted(r[0] for r in rows_both) == [ids[0]]

    assert db.fetch_prune_candidates(min_salience=None, cutoff=None) == []

    to_restore = rows
    for idx, *_ in to_restore:
        db.delete(idx)
        assert db.get(idx) is None

    db.restore_rows(to_restore)
    for row in to_restore:
        idx = row[0]
        value, key, _, _ = db.get(idx)
        assert value == values[ids.index(idx)]
        assert np.allclose(key, keys[ids.index(idx)])

    db.restore_salience(prev)
    for idx, s in zip(ids, saliences):
        _, _, _, restored = db.get(idx)
        assert restored == pytest.approx(s)
