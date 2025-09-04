from omegaconf import OmegaConf

from hippo_eval.bench import _init_modules
from hippo_eval.eval.harness import Task, _run_replay
from hippo_mem.common.telemetry import gate_registry


def test_gate_counters_increment_on_replay() -> None:
    gate_registry.reset()
    cfg = OmegaConf.create(
        {
            "memory": {
                "episodic": {"gate": {"enabled": True, "tau": 0.0}},
                "relational": {"gate": {"enabled": True}},
                "spatial": {"gate": {"enabled": True}},
            }
        }
    )
    modules = _init_modules(cfg.memory, {})
    tasks = [Task(prompt="p1", answer="a"), Task(prompt="p2", answer="b")]
    _run_replay(cfg, modules, tasks)
    assert gate_registry.get("episodic").attempts > 0
    assert gate_registry.get("relational").attempts > 0
    assert gate_registry.get("spatial").attempts > 0
