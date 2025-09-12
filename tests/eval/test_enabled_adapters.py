# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from types import SimpleNamespace

from omegaconf import OmegaConf

from hippo_eval.eval.adapters import enabled_adapters


def test_enabled_adapters_handles_aliases() -> None:
    cfg = SimpleNamespace(memory=["hei_nw", "sgc_rss", "smpd"])
    adapters = enabled_adapters(cfg)
    assert set(adapters) == {"episodic", "relational", "spatial"}


def test_enabled_adapters_handles_dict_cfg() -> None:
    cfg = OmegaConf.create({"memory": {"episodic": {}, "relational": {}}})
    adapters = enabled_adapters(cfg)
    assert set(adapters) == {"episodic", "relational"}


def test_enabled_adapters_empty_when_none() -> None:
    cfg = SimpleNamespace(memory=None)
    assert enabled_adapters(cfg) == {}
