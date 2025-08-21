import sqlite3

from hippo_mem.common.sqlite import SQLiteExecMixin


class Dummy(SQLiteExecMixin):
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")


def test_exec_fetch_one_all_and_commit():
    db = Dummy()
    db._exec("CREATE TABLE t (val TEXT)")
    assert not db.conn.in_transaction

    db._exec("INSERT INTO t (val) VALUES (?)", ("a",))
    assert not db.conn.in_transaction

    row = db._exec("SELECT val FROM t WHERE val=?", ("a",), fetch="one")
    assert row == ("a",)
    assert not db.conn.in_transaction

    db._exec("INSERT INTO t (val) VALUES (?)", ("b",))
    rows = db._exec("SELECT val FROM t ORDER BY val", fetch="all")
    assert rows == [("a",), ("b",)]
    assert not db.conn.in_transaction


def test_exec_unsupported_fetch_returns_none():
    db = Dummy()
    db._exec("CREATE TABLE t (val TEXT)")
    result = db._exec("SELECT val FROM t", fetch="invalid")
    assert result is None
    assert not db.conn.in_transaction
