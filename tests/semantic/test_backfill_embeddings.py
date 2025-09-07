import numpy as np

from hippo_mem.relational.kg import KnowledgeGraph


def test_backfills_missing_embeddings(tmp_path) -> None:
    db = tmp_path / "kg.sqlite"
    kg = KnowledgeGraph(db_path=str(db))
    kg.backend.exec("INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)", ("A", None))
    kg.backend.exec("INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)", ("B", None))
    kg.backend.exec(
        "INSERT INTO edges(src, relation, dst, context, time, conf, provenance, embedding) VALUES (?,?,?,?,?,?,?,?)",
        ("A", "rel", "B", "ctx", None, 1.0, None, None),
    )
    kg2 = KnowledgeGraph(db_path=str(db))
    assert float(np.sum(np.abs(kg2.node_embeddings["A"]))) > 0
    edge = next(iter(kg2.graph["A"]["B"].values()))
    assert edge.get("embedding") is not None
    node_row = kg2.backend.exec("SELECT embedding FROM nodes WHERE name=?", ("A",), fetch="one")
    assert node_row[0] is not None
