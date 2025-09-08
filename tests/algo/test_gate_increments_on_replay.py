from omegaconf import OmegaConf

from hippo_eval.eval.adapters import enabled_adapters
from hippo_eval.eval.harness import _run_replay
from hippo_eval.eval.types import Task
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
    adapters = enabled_adapters(cfg)
    modules = {name: adapter.build(cfg) for name, adapter in adapters.items()}
    tasks = [
        Task(prompt="p1", answer="a", fact="Alice likes Bob."),
        Task(prompt="p2", answer="b", fact="Carol hates Dan."),
    ]
    _run_replay(cfg, modules, tasks)
    assert gate_registry.get("episodic").attempts > 0
    assert gate_registry.get("relational").attempts > 0
    assert gate_registry.get("spatial").attempts > 0
