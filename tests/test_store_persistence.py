import json

import numpy as np
import pytest

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.episodic.types import TraceValue
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.place_graph import PlaceGraph


def test_episodic_save_load(tmp_path) -> None:
    store = EpisodicStore(dim=4)
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
    store.write(vec1, TraceValue(provenance="e1"))
    store.write(vec2, TraceValue(provenance="e2"))

    trace1 = store.recall(vec1, k=1)[0]
    trace2 = store.recall(vec2, k=1)[0]
    store.save(tmp_path, "s1", replay_samples=1)

    new_store = EpisodicStore(dim=4)
    new_store.load(tmp_path, "s1")
    res1 = new_store.recall(vec1, k=1)[0]
    res2 = new_store.recall(vec2, k=1)[0]

    assert res1.value.provenance == "e1"
    assert res2.value.provenance == "e2"
    assert res1.ts == pytest.approx(trace1.ts)
    assert res2.ts == pytest.approx(trace2.ts)


def test_relational_save_load(tmp_path) -> None:
    kg = KnowledgeGraph()
    kg.upsert("alice", "knows", "bob", "ctx1", time="t1", provenance=1)
    kg.upsert("bob", "likes", "carol", "ctx2", time="t2", provenance=2)
    kg.save(tmp_path, "r1", replay_samples=1)

    kg2 = KnowledgeGraph()
    kg2.load(tmp_path, "r1")

    edge_ab = kg2.graph.get_edge_data("alice", "bob")
    attrs_ab = next(iter(edge_ab.values()))
    assert attrs_ab["relation"] == "knows"
    assert attrs_ab["context"] == "ctx1"
    assert attrs_ab["time"] == "t1"
    assert attrs_ab["provenance"] == 1

    edge_bc = kg2.graph.get_edge_data("bob", "carol")
    attrs_bc = next(iter(edge_bc.values()))
    assert attrs_bc["relation"] == "likes"
    assert attrs_bc["context"] == "ctx2"
    assert attrs_bc["time"] == "t2"
    assert attrs_bc["provenance"] == 2


def test_spatial_save_load(tmp_path) -> None:
    pg = PlaceGraph()
    pg.observe("a")
    pg.observe("b")
    pg.observe("c")
    coord_a = pg.encoder._cache["a"].coord
    coord_b = pg.encoder._cache["b"].coord
    coord_c = pg.encoder._cache["c"].coord
    a_id = pg._context_to_id["a"]
    b_id = pg._context_to_id["b"]
    edge_last_seen = pg.graph[a_id][b_id].last_seen
    pg.save(tmp_path, "p1", replay_samples=1)

    pg2 = PlaceGraph()
    pg2.load(tmp_path, "p1")
    a2_id = pg2._context_to_id["a"]
    b2_id = pg2._context_to_id["b"]
    assert pg2.encoder._cache["a"].coord == coord_a
    assert pg2.encoder._cache["b"].coord == coord_b
    assert pg2.encoder._cache["c"].coord == coord_c
    assert pg2.graph[a2_id][b2_id].last_seen == edge_last_seen


@pytest.mark.parametrize(
    "factory,writer,data_file,meta_schema",
    [
        (
            lambda: EpisodicStore(dim=2),
            lambda s: s.write(np.ones(2, dtype=np.float32), TraceValue(provenance="p")),
            "episodic.jsonl",
            "episodic.store_meta.v1",
        ),
        (
            KnowledgeGraph,
            lambda kg: kg.upsert("a", "rel", "b", "ctx"),
            "kg.jsonl",
            "relational.store_meta.v1",
        ),
        (
            PlaceGraph,
            lambda g: (g.observe("a"), g.observe("b")),
            "spatial.jsonl",
            "spatial.store_meta.v1",
        ),
    ],
)
def test_store_meta(tmp_path, factory, writer, data_file, meta_schema) -> None:
    store = factory()
    writer(store)
    store.save(tmp_path, "stub", replay_samples=0)
    stub_dir = tmp_path / "stub"
    meta = json.loads((stub_dir / "store_meta.json").read_text())
    assert meta["schema"] == meta_schema
    assert meta["source"] == "stub"
    assert meta["replay_samples"] == 0
    expected_exists = data_file in {"kg.jsonl", "spatial.jsonl"}
    path = stub_dir / data_file
    if expected_exists:
        assert path.exists()
    else:
        assert not path.exists()

    store.save(tmp_path, "real", replay_samples=1)
    real_dir = tmp_path / "real"
    meta2 = json.loads((real_dir / "store_meta.json").read_text())
    assert meta2["source"] == "replay"
    assert meta2["replay_samples"] == 1
    assert (real_dir / data_file).exists()
