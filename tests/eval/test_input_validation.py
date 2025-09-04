import pytest
from omegaconf import OmegaConf

from hippo_eval.eval import harness


def test_reject_quoted_presets():
    cfg = OmegaConf.create({"run_matrix": True, "presets": "[a,b]", "run_id": "test"})
    with pytest.raises(TypeError, match="presets must be a list"):
        harness.main(cfg)


def test_accept_list_presets(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(harness, "evaluate_matrix", lambda *a, **k: calls.append(True))
    monkeypatch.setattr(harness, "_load_preset", lambda cfg: cfg)
    monkeypatch.setattr(harness, "_apply_model_defaults", lambda cfg: cfg)
    cfg = OmegaConf.create(
        {"run_matrix": True, "presets": ["foo"], "model": "dummy", "run_id": "test"}
    )
    harness.main(cfg)
    assert calls
