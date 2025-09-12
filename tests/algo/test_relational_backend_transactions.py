# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import sqlite3

import pytest

from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph


def test_transaction_rollback_and_commit() -> None:
    """Conflicting upserts trigger rollback while commits persist."""
    kg = KnowledgeGraph()
    conn = kg.backend.conn

    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO edges(id, src, relation, dst, context, time, conf, provenance, embedding) VALUES (1,'A','r','B','ctx',NULL,1.0,0,'[]')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO edges(id, src, relation, dst, context, time, conf, provenance, embedding) VALUES (1,'C','r','D','ctx',NULL,1.0,0,'[]')"
        )
        conn.commit()
    conn.rollback()
    assert kg.backend.exec("SELECT COUNT(*) FROM edges", fetch="one")[0] == 0

    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO edges(id, src, relation, dst, context, time, conf, provenance, embedding) VALUES (1,'A','r','B','ctx',NULL,1.0,0,'[]')"
    )
    conn.commit()
    kg.backend.load(kg.graph, kg.node_embeddings)
    assert kg.graph.has_edge("A", "B")


def test_duplicate_edge_aggregation() -> None:
    """Second ingestion of same tuple boosts confidence instead of new edge."""
    gate = RelationalGate()
    kg = KnowledgeGraph(gate=gate, config={"schema_threshold": 0.1})
    kg.schema_index.add_schema("rel", "rel")
    tup1 = ("A", "rel", "B", "ctx", "0", 0.3, 0)
    kg.ingest(tup1)
    edge_id = next(iter(kg.graph["A"]["B"]))
    tup2 = ("A", "rel", "B", "ctx", "1", 0.2, 1)
    kg.ingest(tup2)
    data = kg.graph["A"]["B"][edge_id]
    assert data["conf"] == pytest.approx(0.5)
    assert data["time"] == "1"
    row = kg.backend.exec("SELECT conf, time FROM edges WHERE id=?", (edge_id,), fetch="one")
    assert row == (0.5, "1")


def test_round_trip_upsert_fetch_delete() -> None:
    """Upsert -> retrieve -> prune keeps graph and DB consistent."""
    kg = KnowledgeGraph()
    kg.upsert("A", "r", "B", "ab", head_embedding=[1.0, 0.0], conf=1.0)
    kg.upsert("B", "r", "C", "bc", head_embedding=[0.5, 0.5], conf=0.4)
    sub = kg.retrieve([1.0, 0.0], k=1, radius=2)
    assert set(sub.nodes()) == {"A", "B", "C"}
    edge_id_bc = next(iter(kg.graph["B"]["C"]))

    kg.prune(min_conf=0.5)
    sub_after = kg.retrieve([1.0, 0.0], k=1, radius=2)
    assert set(sub_after.nodes()) == {"A", "B"}
    assert kg.backend.exec("SELECT id FROM edges WHERE id=?", (edge_id_bc,), fetch="one") is None
    assert kg.backend.exec("SELECT COUNT(*) FROM edges", fetch="one")[0] == 1
