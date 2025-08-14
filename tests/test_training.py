"""Smoke tests for the LoRA/QLoRA training script."""

from types import SimpleNamespace

from scripts.train_lora import TrainConfig, parse_args, train


def test_parse_args_dry_run() -> None:
    """Parsing CLI style overrides returns a config object."""

    cfg = parse_args(["dry_run=true"])
    assert cfg.dry_run is True


def test_train_dry_run_skips_dataset(monkeypatch) -> None:
    """``train`` short circuits before hitting the dataset when dry running."""

    # Provide lightweight stand‑ins for the model and tokenizer to avoid network
    # calls during the test.
    def fake_loader(_cfg: TrainConfig):  # pragma: no cover - trivial helper
        model = SimpleNamespace(config=SimpleNamespace(use_cache=False))
        model.gradient_checkpointing_enable = lambda: None
        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        return model, tokenizer

    def _raise(*_args, **_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("dataset should not be loaded during dry run")

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr("scripts.train_lora.load_dataset", _raise)

    cfg = TrainConfig(dry_run=True)
    train(cfg)
