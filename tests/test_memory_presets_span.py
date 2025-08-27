from pathlib import Path

import yaml


def _assert_preset(p: Path) -> None:
    data = yaml.safe_load(p.read_text())
    assert data["use_chat_template"] is True
    assert "shortest span" in data["system_prompt"]
    assert int(data["max_new_tokens"]) <= 8
    # Retrieval must stay enabled
    assert data["retrieval"]["enabled"] is True
    # Gating toggles remain enabled for memory presets
    mem = data["memory"]
    assert isinstance(mem, dict) and len(mem) == 1
    ((_, cfg),) = mem.items()
    assert cfg["gate"]["enabled"] is True


def test_hei_nw_preset_fields() -> None:
    _assert_preset(Path("configs/eval/memory/hei_nw.yaml"))


def test_sgc_rss_preset_fields() -> None:
    _assert_preset(Path("configs/eval/memory/sgc_rss.yaml"))


def test_smpd_preset_fields() -> None:
    _assert_preset(Path("configs/eval/memory/smpd.yaml"))
