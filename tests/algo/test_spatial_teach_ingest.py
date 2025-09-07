from types import SimpleNamespace

from omegaconf import OmegaConf

from hippo_eval.eval.adapters.spatial import SpatialEvalAdapter
from hippo_mem.common.gates import GateCounters
from hippo_mem.spatial.gating import SpatialGate
from hippo_mem.spatial.map import PlaceGraph


def test_spatial_teach_ingest_builds_graph() -> None:
    graph = PlaceGraph()
    gate = SpatialGate(block_threshold=2.0)
    modules = {"map": graph, "gate": gate}
    adapter = SpatialEvalAdapter()
    cfg = OmegaConf.create({})
    gc = GateCounters()
    item = SimpleNamespace(prompt="Start (0,0)", answer="UR", context_key=None)
    adapter.teach(cfg, modules, item, dry_run=False, gc=gc, suite="spatial_multi")
    assert gc.accepted > 0
    assert len(graph.graph) >= 2
