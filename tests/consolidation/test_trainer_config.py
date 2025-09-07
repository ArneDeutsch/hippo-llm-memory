import logging

import pytest
from torch import nn

import hippo_mem.consolidation.trainer as trainer_mod


def test_deep_update_merges_nested() -> None:
    base = {"a": {"b": 1}, "c": 2}
    upd = {"a": {"d": 3}, "e": 4}
    result = trainer_mod._deep_update(base, upd)
    assert result["a"] == {"b": 1, "d": 3}
    assert result["e"] == 4


def test_configure_lora_warns_on_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Dummy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm = nn.Linear(1, 1)

    obj = trainer_mod.ConsolidationTrainer.__new__(trainer_mod.ConsolidationTrainer)
    obj.model = Dummy()
    obj.lora_hash = ""
    monkeypatch.setattr(trainer_mod, "default_target_modules", lambda model: ["lm"])
    monkeypatch.setattr(trainer_mod, "get_peft_model", lambda model, cfg: model)
    cfg = {"targets": ["missing"], "rank": 1, "alpha": 1, "dropout": 0.0}
    with caplog.at_level(logging.WARNING):
        obj.configure_lora(cfg)
        assert "target modules" in caplog.text
