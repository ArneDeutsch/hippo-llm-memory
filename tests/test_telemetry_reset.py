from omegaconf import OmegaConf

from hippo_eval.harness import build_runner, run_suite
from hippo_mem.common.telemetry import registry


def test_telemetry_resets_between_runs() -> None:
    registry.reset()
    cfg_mem = OmegaConf.create(
        {
            "suite": "episodic",
            "n": 5,
            "seed": 1337,
            "preset": "configs/eval/memory/hei_nw.yaml",
            "model": "models/tiny-gpt2",
        }
    )
    run_suite(build_runner(cfg_mem))
    first = registry.get("episodic").snapshot()["requests"]
    assert first > 0

    cfg_base = OmegaConf.create(
        {
            "suite": "episodic",
            "n": 5,
            "seed": 1337,
            "preset": "configs/eval/baselines/core.yaml",
            "model": "models/tiny-gpt2",
        }
    )
    run_suite(build_runner(cfg_base))
    second = registry.get("episodic").snapshot()["requests"]
    assert second == 0
