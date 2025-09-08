from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from hippo_eval.eval.adapters import EpisodicEvalAdapter
from hippo_mem.common.gates import GateCounters
from hippo_mem.episodic.gating import WriteGate
from hippo_mem.episodic.store import EpisodicStore


def test_episodic_teach_ingest_populates_store() -> None:
    store = EpisodicStore(dim=8, k_wta=2)
    gate = WriteGate(tau=0.0)
    modules = {"store": store, "gate": gate}
    adapter = EpisodicEvalAdapter()
    cfg = OmegaConf.create({})
    gc = GateCounters()
    for i in range(5):
        item = SimpleNamespace(
            fact=f"Person {i} went to Place {i}.",
            context_key=f"ep/{i:02d}",
            prompt="",
            answer="",
        )
        adapter.teach(cfg, modules, item, dry_run=False, gc=gc, suite="episodic_cross_mem")
    keys = store.keys()
    nonzero = sum(np.linalg.norm(k) > 0 for k in keys)
    assert nonzero / len(keys) >= 0.9
    for _idx, val, _k, _ts, _sal in store.persistence.all():
        assert val.provenance == "teach"
        assert val.context_key is not None
    size, _diag = adapter.store_size(modules)
    assert size >= len(keys)


def test_episodic_teach_ablation_skips_writes() -> None:
    store = EpisodicStore(dim=8, k_wta=2)
    gate = WriteGate(tau=1.0)
    modules = {"store": store, "gate": gate}
    adapter = EpisodicEvalAdapter()
    cfg = OmegaConf.create({})
    gc = GateCounters()
    item = SimpleNamespace(fact="A went to B.", context_key="ep/00", prompt="", answer="")
    adapter.teach(cfg, modules, item, dry_run=False, gc=gc, suite="episodic_cross_mem")
    size, _diag = adapter.store_size(modules)
    assert size == 0
    assert gc.skipped >= 1
