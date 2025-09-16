# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from omegaconf import OmegaConf

from hippo_eval.harness import build_runner
from hippo_mem.common.telemetry import registry
from hippo_mem.testing import FAKE_MODEL_ID


def test_telemetry_resets_between_runs() -> None:
    registry.reset()
    cfg_mem = OmegaConf.create(
        {
            "suite": "episodic_cross_mem",
            "n": 5,
            "seed": 1337,
            "preset": "configs/eval/memory/hei_nw.yaml",
            "model": FAKE_MODEL_ID,
        }
    )
    build_runner(cfg_mem).run()
    first = registry.get("episodic").snapshot()["requests"]
    assert first > 0

    cfg_base = OmegaConf.create(
        {
            "suite": "episodic_cross_mem",
            "n": 5,
            "seed": 1337,
            "preset": "configs/eval/baselines/core.yaml",
            "model": FAKE_MODEL_ID,
        }
    )
    build_runner(cfg_base).run()
    second = registry.get("episodic").snapshot()["requests"]
    assert second == 0
