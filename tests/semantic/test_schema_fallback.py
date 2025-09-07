from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.relational.schema import SchemaIndex


def test_fast_track_inserts_when_no_schemas() -> None:
    kg = KnowledgeGraph()
    kg.schema_index = SchemaIndex(threshold=0.5, add_defaults=False)
    tup = ("alice", "likes", "bob", "ctx", None, 1.0, 0)
    inserted = kg.schema_index.fast_track(tup, kg)
    assert inserted
    assert kg.graph.number_of_edges() == 1
