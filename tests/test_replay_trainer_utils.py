import pytest
import torch

from hippo_mem.consolidation.trainer import _compute_kl, _format_records, load_config


def test_format_records_handles_prompt_and_q() -> None:
    records = [
        {"prompt": "p", "answer": "a"},
        {"q": "x", "a": "y"},
        {"prompt": "only"},
    ]
    texts, kept = _format_records(records)
    assert texts == ["p\na", "x\ny", "only"]
    assert kept == records


def test_format_records_skips_empty_entries() -> None:
    records = [
        {"prompt": "", "answer": ""},
        {"q": "x", "a": "y"},
    ]
    texts, kept = _format_records(records)
    assert texts == ["x\ny"]
    assert kept == [{"q": "x", "a": "y"}]


def test_compute_kl_zero_for_identical_logits() -> None:
    logits = torch.tensor([[0.5, 1.0]])
    kl = _compute_kl(logits, logits)
    assert kl.item() == pytest.approx(0.0, abs=1e-6)


def test_load_config_overrides(tmp_path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("train:\n  steps: 1\n  batch_size: 2\n")
    cfg = load_config(str(cfg_file))
    assert cfg["train"]["steps"] == 1
    assert cfg["train"]["batch_size"] == 2
    # ensure defaults preserved
    assert "rank" in cfg["peft"]
