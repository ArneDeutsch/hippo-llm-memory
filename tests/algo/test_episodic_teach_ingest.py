from types import SimpleNamespace

import numpy as np

from hippo_eval.eval.harness import _ingest_episodic
from hippo_mem.common.gates import GateCounters
from hippo_mem.episodic.gating import WriteGate
from hippo_mem.episodic.store import EpisodicStore


def test_episodic_teach_ingest_populates_store() -> None:
    store = EpisodicStore(dim=8, k_wta=2)
    gate = WriteGate(tau=0.0)
    modules = {"episodic": {"store": store}}
    gc = GateCounters()
    for i in range(5):
        item = SimpleNamespace(
            fact=f"Person {i} went to Place {i}.",
            context_key=f"ep/{i:02d}",
            prompt="",
            answer="",
        )
        _ingest_episodic(item, modules, gate, gc, "episodic_cross_mem", False)
    keys = store.keys()
    nonzero = sum(np.linalg.norm(k) > 0 for k in keys)
    assert nonzero / len(keys) >= 0.9
    for _idx, val, _k, _ts, _sal in store.persistence.all():
        assert val.provenance == "teach"
        assert val.context_key is not None
