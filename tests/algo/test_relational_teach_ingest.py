from types import SimpleNamespace

from omegaconf import OmegaConf

from hippo_eval.eval.adapters.relational import RelationalEvalAdapter
from hippo_mem.common.gates import GateCounters
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph


def test_relational_teach_ingest_populates_kg() -> None:
    kg = KnowledgeGraph()
    adapter = RelationalEvalAdapter()
    modules = {"kg": kg, "gate": RelationalGate()}
    gc = GateCounters()
    cfg = OmegaConf.create({})
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
        adapter.teach(cfg, modules, it, dry_run=False, gc=gc, suite="test")
    kg.schema_index.flush(kg)
    assert kg.graph.number_of_nodes() > 0
    assert kg.graph.number_of_edges() > 0
    assert kg.schema_index.episodic_buffer == []
