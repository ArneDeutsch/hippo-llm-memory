import json
import sys

import numpy as np

import scripts.validate_store as vs
from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.episodic.types import TraceValue
from hippo_mem.eval.harness import _store_sizes
from hippo_mem.relational.kg import KnowledgeGraph
from hippo_mem.spatial.map import PlaceGraph


def run_validator(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["validate_store", *args])
    vs.main()


def test_store_accounting(tmp_path, monkeypatch):
    store = EpisodicStore(4)
    kg = KnowledgeGraph()
    g = PlaceGraph()

    store.write(np.ones(4, dtype=np.float32), TraceValue(provenance="p"))
    kg.upsert("a", "rel", "b", "ctx")
    g.observe("a")
    g.observe("b")

    modules = {"episodic": {"store": store}, "relational": {"kg": kg}, "spatial": {"map": g}}
    sizes, diags = _store_sizes(modules)
    assert sizes == {"episodic": 1, "relational": 1, "spatial": 4}
    assert diags["relational"]["nodes_added"] == 2
    assert diags["spatial"]["writes"] == 2

    store_dir = tmp_path / "runs" / "foo" / "stores"
    algo_dir = store_dir / "hei_nw"
    session_id = "hei_foo"
    store.save(str(algo_dir), session_id)
    kg.save(str(algo_dir), session_id)
    g.save(str(algo_dir), session_id)

    metrics = {"store": {"size": sum(sizes.values()), "per_memory": sizes, "diagnostics": diags}}
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_ID", "foo")
    for kind in ("episodic", "kg", "spatial"):
        run_validator(
            monkeypatch, ["--algo", "hei_nw", "--kind", kind, "--metrics", str(metrics_path)]
        )
