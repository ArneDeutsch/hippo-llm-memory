import json
import sqlite3

from hippo_mem.relational.backend import SQLiteBackend
from hippo_mem.relational.kg import KnowledgeGraph


def test_relational_restore_ignores_duplicate_edges(tmp_path) -> None:
    """Loading a store with duplicate edges inserts only one row."""
    store = tmp_path / "stores" / "hei_nw" / "sess" / "relational.jsonl"
    store.parent.mkdir(parents=True)
    edge = {
        "type": "edge",
        "id": 1,
        "src": "a",
        "relation": "R",
        "dst": "b",
        "context": "ctx",
        "time": None,
        "conf": 1.0,
        "provenance": 0,
        "embedding": None,
    }
    with store.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(edge) + "\n")
        fh.write(json.dumps(edge) + "\n")

    db_path = tmp_path / "kg.sqlite"
    kg = KnowledgeGraph(backend=SQLiteBackend(str(db_path)))
    kg.load(str(tmp_path / "stores" / "hei_nw"), "sess")

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM edges WHERE id=1")
        assert cur.fetchone()[0] == 1
