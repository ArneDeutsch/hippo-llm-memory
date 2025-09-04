from pathlib import Path

import yaml


def test_span_short_preset_fields():
    p = Path("configs/eval/baselines/span_short.yaml")
    data = yaml.safe_load(p.read_text())
    assert data["use_chat_template"] is True
    assert "shortest span" in data["system_prompt"]
    assert int(data["max_new_tokens"]) <= 8
    # Baseline toggles remain off
    assert data["retrieval"]["enabled"] is False
    assert data["gating_enabled"] is False


def test_longctx_preset_fields() -> None:
    p = Path("configs/eval/baselines/longctx.yaml")
    data = yaml.safe_load(p.read_text())
    assert data["retrieval"]["enabled"] is False
    assert data["long_context"]["enabled"] is True
    assert data["memory"] is None
    assert data["gating_enabled"] is False


def test_rag_preset_fields() -> None:
    p = Path("configs/eval/baselines/rag.yaml")
    data = yaml.safe_load(p.read_text())
    assert data["retrieval"]["enabled"] is True
    assert data["long_context"]["enabled"] is False
    assert data["memory"] is None
    assert data["gating_enabled"] is False
