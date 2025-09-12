# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from omegaconf import OmegaConf

from hippo_eval.eval.harness import _enforce_guardrails


def test_baseline_ceiling_guard_raises() -> None:
    cfg = OmegaConf.create({"baseline_ceiling": 0.2})
    metrics = {"em_raw": 0.5}
    try:
        _enforce_guardrails(cfg, metrics, None, {}, has_memory=False)
    except RuntimeError as err:
        assert "baseline EM" in str(err)
    else:  # pragma: no cover - should not happen
        raise AssertionError("baseline guard did not trigger")


def test_baseline_ceiling_guard_override() -> None:
    cfg = OmegaConf.create({"baseline_ceiling": 0.2, "allow_baseline_high": True})
    metrics = {"em_raw": 0.5}
    _enforce_guardrails(cfg, metrics, None, {}, has_memory=False)
