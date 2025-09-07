from types import SimpleNamespace

from hippo_eval.eval.harness import _ingest_semantic
from hippo_mem.common.gates import GateCounters
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph


def test_relational_teach_ingest_populates_kg() -> None:
    kg = KnowledgeGraph()
    modules = {"relational": {"kg": kg}}
    gate = RelationalGate()
    gc = GateCounters()
    items = [
        SimpleNamespace(
            fact="Alice bought a book at StoreA.",
            context_key="sem/0",
            facts=None,
            prompt="",
            answer="",
        ),
        SimpleNamespace(
            fact="StoreA is in Paris.", context_key="sem/0", facts=None, prompt="", answer=""
        ),
    ]
    for it in items:
        _ingest_semantic(it, modules, gate, gc, False)
    kg.schema_index.flush(kg)
    assert kg.graph.number_of_nodes() > 0
    assert kg.graph.number_of_edges() > 0
    assert kg.schema_index.episodic_buffer == []
