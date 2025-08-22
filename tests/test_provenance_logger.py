from hippo_mem.common.provenance import ProvenanceLogger
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph


def test_provenance_written(tmp_path):
    logger = ProvenanceLogger(tmp_path)
    kg = KnowledgeGraph(gate=RelationalGate(), provenance=logger)
    kg.ingest(("A", "likes", "B", "ctx", None, 0.9, 0))
    path = tmp_path / "provenance.ndjson"
    assert path.exists()
    lines = path.read_text().splitlines()
    assert len(lines) > 0
