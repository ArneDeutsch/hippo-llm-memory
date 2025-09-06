import numpy as np
import pytest

from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.retrieval.embed import embed_text


@pytest.fixture
def kg() -> KnowledgeGraph:
    return KnowledgeGraph()


def test_kg_auto_embeddings(kg: KnowledgeGraph) -> None:
    kg.upsert("Carol", "bought_at", "StoreB", "Carol bought at StoreB")
    kg.upsert("StoreB", "located_in", "Berlin", "StoreB located in Berlin")

    for name in ("Carol", "StoreB", "Berlin"):
        vec = kg.node_embeddings.get(name)
        assert vec is not None
        assert float(np.sum(np.abs(vec))) > 0

    edge1 = next(iter(kg.graph["Carol"]["StoreB"].values()))
    edge2 = next(iter(kg.graph["StoreB"]["Berlin"].values()))
    for edge in (edge1, edge2):
        vec = edge.get("embedding")
        assert vec is not None
        assert float(np.sum(np.abs(vec))) > 0

    query_vec = embed_text("Carol bought apple in which city?")
    sub = kg.retrieve(query_vec, k=4, radius=2)
    assert {"Carol", "StoreB", "Berlin"} <= set(sub.nodes())
    rels = {d["relation"] for _, _, d in sub.edges(data=True)}
    assert {"bought_at", "located_in"} <= rels
