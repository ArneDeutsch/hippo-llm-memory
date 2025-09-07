from types import SimpleNamespace

from hippo_eval.eval.harness import _ingest_spatial
from hippo_mem.common.gates import GateCounters
from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.map import PlaceGraph


def test_spatial_teach_ingest_builds_graph() -> None:
    graph = PlaceGraph()
    gate = SpatialGate(block_threshold=2.0)
    modules = {"spatial": {"map": graph}}
    gc = GateCounters()
    item = SimpleNamespace(prompt="Start (0,0)", answer="UR", context_key=None)
    _ingest_spatial(item, modules, gate, gc, False)
    assert gc.accepted > 0
    assert len(graph.graph) >= 2
