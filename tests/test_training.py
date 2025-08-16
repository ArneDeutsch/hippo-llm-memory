"""Smoke tests for the LoRA/QLoRA training script."""

import subprocess
import sys
from types import SimpleNamespace

from scripts.train_lora import (
    TrainConfig,
    _load_model_and_tokenizer,
    parse_args,
    train,
)


def test_parse_args_dry_run() -> None:
    """Parsing CLI style overrides returns a config object."""

    cfg = parse_args(["dry_run=true"])
    assert cfg.dry_run is True


def test_train_sets_seeds(monkeypatch) -> None:
    """Setting the seed propagates to RNG libraries."""

    calls: dict[str, int] = {}

    monkeypatch.setattr("scripts.train_lora.random.seed", lambda v: calls.setdefault("py", v))
    monkeypatch.setattr("scripts.train_lora.np.random.seed", lambda v: calls.setdefault("np", v))
    monkeypatch.setattr(
        "scripts.train_lora.torch.manual_seed", lambda v: calls.setdefault("torch", v)
    )
    monkeypatch.setattr("scripts.train_lora.torch.cuda.is_available", lambda: False)

    def fake_loader(_cfg: TrainConfig):
        model = SimpleNamespace(config=SimpleNamespace(use_cache=False, hidden_size=8))
        model.gradient_checkpointing_enable = lambda: None
        model.set_attn_implementation = lambda *_a, **_k: None
        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        return model, tokenizer

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)

    cfg = TrainConfig(dry_run=True, seed=123)
    train(cfg)

    assert calls == {"py": 123, "np": 123, "torch": 123}


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


def test_loads_real_model_offline(monkeypatch) -> None:
    """Model and tokenizer load locally without network access."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_MODEL_PATH", "models/tiny-gpt2")

    cfg = TrainConfig(dry_run=True)
    model, tokenizer = _load_model_and_tokenizer(cfg)

    assert model is not None
    assert tokenizer is not None


def test_adapter_ablation_flags(monkeypatch) -> None:
    """Hydra flags toggle adapters independently."""

    created: list[str] = []

    def fake_loader(_cfg: TrainConfig):
        model = SimpleNamespace(config=SimpleNamespace(use_cache=False, hidden_size=8))
        model.gradient_checkpointing_enable = lambda: None
        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        return model, tokenizer

    class DummyStore:
        def __init__(self, _hidden, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:  # pragma: no cover - trivial
            return "store"

    class DummyKG:
        def __init__(self, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:  # pragma: no cover - trivial
            return "kg"

    class DummyMap:
        def __init__(self, path_integration=False, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:  # pragma: no cover - trivial
            return "map"

    def fake_epi(cfg):
        created.append("episodic")
        return SimpleNamespace()

    def fake_rel():
        created.append("relational")
        return SimpleNamespace()

    def fake_spat(cfg):
        created.append("spatial")
        return SimpleNamespace()

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr("scripts.train_lora.EpisodicStore", DummyStore)
    monkeypatch.setattr("scripts.train_lora.KnowledgeGraph", DummyKG)
    monkeypatch.setattr("scripts.train_lora.PlaceGraph", DummyMap)
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)
    monkeypatch.setattr("scripts.train_lora.EpisodicAdapter", fake_epi)
    monkeypatch.setattr("scripts.train_lora.RelationalAdapter", fake_rel)
    monkeypatch.setattr("scripts.train_lora.SpatialAdapter", fake_spat)

    cfg = parse_args(
        [
            "dry_run=true",
            "episodic.enabled=false",
            "relational=true",
            "spatial.enabled=false",
            "replay.enabled=false",
        ]
    )
    train(cfg)
    assert created == ["relational"]


def test_replay_flag_controls_scheduler(monkeypatch) -> None:
    """Replay toggle skips scheduler and worker when disabled."""

    calls: list[str] = []

    def fake_loader(_cfg: TrainConfig):
        model = SimpleNamespace(config=SimpleNamespace(use_cache=False, hidden_size=8))
        model.gradient_checkpointing_enable = lambda: None
        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        return model, tokenizer

    class DummyStore:
        def __init__(self, _hidden, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:  # pragma: no cover - trivial
            return "store"

    class DummyKG:
        def __init__(self, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:  # pragma: no cover - trivial
            return "kg"

    class DummyMap:
        def __init__(self, path_integration=False, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:  # pragma: no cover - trivial
            return "map"

    def fake_scheduler(*_a, **_k):
        calls.append("scheduler")

        class Dummy:
            def add_trace(self, *a, **k):
                pass

            def log_status(self) -> str:  # pragma: no cover - trivial
                return "sched"

        return Dummy()

    def fake_worker(*_a, **_k):
        calls.append("worker")

        class Dummy:
            def start(self):
                pass

            def stop(self):
                pass

            def join(self, timeout=None):  # pragma: no cover - trivial
                pass

            def log_status(self) -> str:  # pragma: no cover - trivial
                return "worker"

        return Dummy()

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr("scripts.train_lora.EpisodicStore", DummyStore)
    monkeypatch.setattr("scripts.train_lora.KnowledgeGraph", DummyKG)
    monkeypatch.setattr("scripts.train_lora.PlaceGraph", DummyMap)
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)
    monkeypatch.setattr("scripts.train_lora.EpisodicAdapter", lambda cfg: SimpleNamespace())
    monkeypatch.setattr("scripts.train_lora.RelationalAdapter", lambda: SimpleNamespace())
    monkeypatch.setattr("scripts.train_lora.SpatialAdapter", lambda cfg: SimpleNamespace())
    monkeypatch.setattr("scripts.train_lora.ReplayScheduler", fake_scheduler)
    monkeypatch.setattr("scripts.train_lora.ConsolidationWorker", fake_worker)

    cfg = parse_args(["dry_run=true", "episodic.enabled=true", "replay.enabled=false"])
    train(cfg)
    assert calls == []

    calls.clear()
    cfg = parse_args(["dry_run=true", "episodic.enabled=true", "replay.enabled=true"])
    train(cfg)
    assert calls == ["scheduler", "worker"]


def test_flash_attention_toggle(monkeypatch) -> None:
    """Setting efficiency.flash_attention uses flash attention kernels if available."""

    called: dict[str, str] = {}

    class DummyModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(use_cache=False, hidden_size=8)
            self.gradient_checkpointing_enable = lambda: None

        def set_attn_implementation(self, name: str) -> None:
            called["impl"] = name

    monkeypatch.setattr(
        "scripts.train_lora.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )
    monkeypatch.setattr(
        "scripts.train_lora.AutoTokenizer.from_pretrained",
        lambda *a, **k: SimpleNamespace(pad_token=None, eos_token="<eos>"),
    )

    cfg = TrainConfig()
    cfg.efficiency.flash_attention = True
    _load_model_and_tokenizer(cfg)

    assert called.get("impl") == "flash_attention_2"


def test_cli_respects_ablation_flags(monkeypatch) -> None:
    """Running the script with ablation flags disables components."""

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_MODEL_PATH", "models/tiny-gpt2")

    cmd = [
        sys.executable,
        "-m",
        "scripts.train_lora",
        "dry_run=true",
        "episodic.enabled=true",
        "replay.enabled=false",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = proc.stdout + proc.stderr
    assert "scheduler=disabled" in output
    assert "worker=disabled" in output


def test_train_respects_hopfield_flag(monkeypatch) -> None:
    """Setting ``episodic.hopfield=false`` bypasses completion calls."""

    class DummyStore:
        instance = None

        def __init__(self, _hidden, config=None) -> None:
            self.config = config or {}
            self.complete_called = False
            DummyStore.instance = self

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> dict:
            return {}

        def complete(self, query, k=1):  # pragma: no cover - should not run
            self.complete_called = True
            return query

    class DummyKG:
        def __init__(self, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> dict:
            return {}

    class DummyMap:
        def __init__(self, path_integration=False, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> dict:
            return {}

    def fake_loader(_cfg: TrainConfig):
        model = SimpleNamespace(config=SimpleNamespace(use_cache=False, hidden_size=8))
        model.gradient_checkpointing_enable = lambda: None
        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        return model, tokenizer

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr("scripts.train_lora.EpisodicStore", DummyStore)
    monkeypatch.setattr("scripts.train_lora.KnowledgeGraph", DummyKG)
    monkeypatch.setattr("scripts.train_lora.PlaceGraph", DummyMap)
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)

    cfg = parse_args(["dry_run=true", "episodic.hopfield=false", "replay.enabled=false"])
    train(cfg)

    store = DummyStore.instance
    assert store is not None
    assert store.config["hopfield"] is False
    assert store.complete_called is False
