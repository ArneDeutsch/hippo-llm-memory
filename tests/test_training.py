"""Smoke tests for the LoRA/QLoRA training script."""

from scripts.train_lora import parse_args


def test_parse_args_dry_run() -> None:
    """Parsing CLI style overrides returns a config object."""

    cfg = parse_args(["dry_run=true"])
    assert cfg.dry_run is True
