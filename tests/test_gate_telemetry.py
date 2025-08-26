from pathlib import Path

from hippo_mem.common import ProvenanceLogger
from hippo_mem.common.telemetry import gate_registry
from hippo_mem.eval.harness import EvalConfig, run_suite
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.map import PlaceGraph
from hippo_mem.training.lora import build_spatial_gate, ingest_spatial_traces


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


def test_gate_counter_invariants() -> None:
    """Gate counters match decisions for relational and spatial inserts."""

    gate_registry.reset()

    kg = KnowledgeGraph(gate=RelationalGate(threshold=0.9))
    kg.schema_index.add_schema("likes", "likes")

    kg.ingest(("A", "likes", "B", "ctx", None, 0.95, 0))
    kg.ingest(("A", "likes", "B", "ctx", None, 0.95, 1))
    kg.ingest(("C", "likes", "D", "ctx", None, 0.1, 2))
    rel = gate_registry.get("relational")
    assert (rel.attempts, rel.inserted, rel.aggregated, rel.routed_to_episodic) == (
        3,
        1,
        1,
        1,
    )
    assert rel.attempts == rel.inserted + rel.aggregated + rel.routed_to_episodic

    records = [{"trajectory": [(0, 0), (1, 0), (0, 0), (0, 0), (0, 0)]}]
    graph = PlaceGraph()
    gate = build_spatial_gate({"enabled": True, "block_threshold": 0.5, "repeat_N": 3})
    ingest_spatial_traces(records, graph, gate)
    spa = gate_registry.get("spatial")
    assert (spa.attempts, spa.inserted, spa.aggregated, spa.blocked_new_edges) == (
        5,
        3,
        1,
        1,
    )
    assert spa.attempts == spa.inserted + spa.aggregated + spa.blocked_new_edges
