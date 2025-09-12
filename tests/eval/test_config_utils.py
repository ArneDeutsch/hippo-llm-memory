# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from omegaconf import OmegaConf

from hippo_eval.eval.config_utils import apply_ablation_flags, merge_memory_shortcuts


def test_merge_memory_shortcuts_preserves_overrides():
    cfg = OmegaConf.create(
        {
            "preset": "memory/hei_nw",
            "memory": {"episodic": {"a": 1}},
            "episodic": {"b": 2},
            "relational": {"c": 3},
        }
    )
    merge_memory_shortcuts(cfg)
    assert "episodic" not in cfg
    assert "relational" not in cfg
    assert cfg.memory.episodic == {"a": 1, "b": 2}
    assert "relational" not in cfg.memory


def test_merge_memory_shortcuts_no_memory_section_when_baseline():
    cfg = OmegaConf.create({"preset": "baselines/core", "episodic": {"a": 1}})
    merge_memory_shortcuts(cfg)
    assert "episodic" not in cfg
    assert cfg.get("memory") in (None, {})


def test_apply_ablation_flags_updates_memory_cfg():
    cfg = OmegaConf.create(
        {
            "memory": {
                "episodic": {"gate": {"enabled": True}, "use_completion": True},
                "relational": {"gate": {"enabled": True}},
                "spatial": {"gate": {"enabled": True}},
            }
        }
    )
    flat_ablate = {
        "episodic.use_gate": False,
        "episodic.use_completion": False,
        "relational.gate.enabled": False,
        "spatial.gate.enabled": False,
    }
    apply_ablation_flags(cfg, flat_ablate)
    mem = cfg.memory
    assert mem.episodic.gate.enabled is False
    assert mem.episodic.use_completion is False
    assert mem.relational.gate.enabled is False
    assert mem.spatial.gate.enabled is False
