from hippo_mem.common.telemetry import registry
from hippo_mem.eval.harness import EvalConfig, run_suite


def test_telemetry_resets_between_runs() -> None:
    registry.reset()
    cfg_mem = EvalConfig(
        suite="episodic",
        n=5,
        seed=1337,
        preset="configs/eval/memory/hei_nw.yaml",
        model="models/tiny-gpt2",
    )
    run_suite(cfg_mem)
    first = registry.get("episodic").snapshot()["requests"]
    assert first > 0

    cfg_base = EvalConfig(
        suite="episodic",
        n=5,
        seed=1337,
        preset="configs/eval/baselines/core.yaml",
        model="models/tiny-gpt2",
    )
    run_suite(cfg_base)
    second = registry.get("episodic").snapshot()["requests"]
    assert second == 0
