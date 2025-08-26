from pathlib import Path

from hippo_mem.common import ProvenanceLogger
from hippo_mem.common.telemetry import gate_registry
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.map import PlaceGraph
from scripts.eval_model import EvalConfig, run_suite
from scripts.train_lora import build_spatial_gate, ingest_spatial_traces


def test_gate_counters_increment() -> None:
    gate_registry.reset()
    kg = KnowledgeGraph(gate=RelationalGate())
    kg.ingest(("A", "likes", "B", "ctx", None, 0.9, 0))
    rel = gate_registry.get("relational")
    assert rel.attempts == 1
    assert rel.inserted == 1

    graph = PlaceGraph()
    records = [{"trajectory": [(0, 0), (1, 0)]}]
    gate = build_spatial_gate({"enabled": True})
    ingest_spatial_traces(records, graph, gate)
    spa = gate_registry.get("spatial")
    assert spa.attempts >= 2
    assert spa.inserted >= 2


def test_provenance_logger_writes(tmp_path: Path) -> None:
    logger = ProvenanceLogger(str(tmp_path))
    kg = KnowledgeGraph(gate=RelationalGate(logger=logger))
    kg.ingest(("A", "likes", "B", "ctx", None, 0.9, 0))
    graph = PlaceGraph()
    records = [{"trajectory": [(0, 0), (1, 0)]}]
    gate = build_spatial_gate({"enabled": True}, logger)
    ingest_spatial_traces(records, graph, gate)
    log_file = tmp_path / "provenance.ndjson"
    assert log_file.exists()
    assert log_file.read_text().strip()


def test_gate_metrics_propagate(tmp_path: Path) -> None:
    gate_registry.reset()
    cfg = EvalConfig(
        suite="episodic",
        n=5,
        seed=1337,
        preset="configs/eval/memory/hei_nw.yaml",
    )
    rows, metrics, _ = run_suite(cfg)
    assert rows
    gates = metrics.get("gates")
    assert gates is not None
    assert "relational" in gates and "spatial" in gates
    assert "accepts" in gates["relational"]
