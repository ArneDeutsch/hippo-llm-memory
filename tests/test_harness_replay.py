from omegaconf import OmegaConf

from hippo_mem.episodic.store import EpisodicStore
from hippo_mem.eval.harness import Task, _run_replay


def test_run_replay_uses_write_gate() -> None:
    cfg = OmegaConf.create({"memory": {"episodic": {"gate": {"tau": 0.0}}}})
    store = EpisodicStore(8)
    modules = {"episodic": {"store": store}}
    tasks = [Task(prompt="p1", answer="a"), Task(prompt="p2", answer="b")]
    count = _run_replay(cfg, modules, tasks)
    assert count == len(tasks)
