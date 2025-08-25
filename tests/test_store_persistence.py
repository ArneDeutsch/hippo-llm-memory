import numpy as np

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.episodic.types import TraceValue
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.place_graph import PlaceGraph


def test_episodic_save_load(tmp_path) -> None:
    store = EpisodicStore(dim=4)
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    store.write(vec, TraceValue(provenance="e1"))
    store.save(tmp_path, "s1")

    new_store = EpisodicStore(dim=4)
    new_store.load(tmp_path, "s1")
    res = new_store.recall(vec, k=1)
    assert res and res[0].value.provenance == "e1"


def test_relational_save_load(tmp_path) -> None:
    kg = KnowledgeGraph()
    kg.upsert("alice", "knows", "bob", "ctx")
    kg.save(tmp_path, "r1")

    kg2 = KnowledgeGraph()
    kg2.load(tmp_path, "r1")
    assert kg2.graph.has_edge("alice", "bob")


def test_spatial_save_load(tmp_path) -> None:
    pg = PlaceGraph()
    pg.observe("a")
    pg.observe("b")
    pg.save(tmp_path, "p1")

    pg2 = PlaceGraph()
    pg2.load(tmp_path, "p1")
    assert pg2.plan("a", "b") == ["a", "b"]
