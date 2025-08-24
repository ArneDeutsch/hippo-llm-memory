"""Smoke tests for the LoRA/QLoRA training script."""

import logging
import subprocess
import sys
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

from hippo_mem.adapters.relational_adapter import RelationalMemoryAdapter
from hippo_mem.common import MemoryTokens
from scripts.train_lora import (
    TrainConfig,
    _init_sft_trainer,
    _load_model_and_tokenizer,
    parse_args,
    train,
)


def test_parse_args_dry_run() -> None:
    """Parsing CLI style overrides returns a config object."""

    cfg = parse_args(["dry_run=true"])
    assert cfg.dry_run is True


def test_train_config_defaults() -> None:
    """Default config uses research-backed LoRA parameters."""

    cfg = TrainConfig()
    assert cfg.gradient_accumulation_steps == 4
    assert cfg.max_steps == 500
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.1
    assert cfg.episodic.lora_r == 16
    assert cfg.spatial.lora_r == 16


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
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters",
        lambda *a, **k: {"target_block": 0, "num_blocks": 1},
    )

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
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters",
        lambda *a, **k: {"target_block": 0, "num_blocks": 1},
    )

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
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters",
        lambda *a, **k: {"target_block": 0, "num_blocks": 1},
    )
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters",
        lambda *a, **k: {"target_block": 0, "num_blocks": 1},
    )
    monkeypatch.setattr("scripts.train_lora.EpisodicAdapter", fake_epi)
    monkeypatch.setattr("scripts.train_lora.RelationalMemoryAdapter", fake_rel)
    monkeypatch.setattr("scripts.train_lora.SpatialAdapter", fake_spat)
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters",
        lambda *a, **k: {"target_block": 0, "num_blocks": 1},
    )

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
    monkeypatch.setattr("scripts.train_lora.RelationalMemoryAdapter", lambda: SimpleNamespace())
    monkeypatch.setattr("scripts.train_lora.SpatialAdapter", lambda cfg: SimpleNamespace())
    monkeypatch.setattr("scripts.train_lora.ReplayScheduler", fake_scheduler)
    monkeypatch.setattr("scripts.train_lora.ConsolidationWorker", fake_worker)
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters",
        lambda *a, **k: {"target_block": 0, "num_blocks": 1},
    )

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


def test_train_respects_mqa_gqa_flag(monkeypatch) -> None:
    """Setting efficiency.mqa_gqa adjusts adapter KV heads."""

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

        def log_status(self) -> str:
            return "store"

    class DummyKG:
        def __init__(self, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:
            return "kg"

    class DummyMap:
        def __init__(self, path_integration=False, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:
            return "map"

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr("scripts.train_lora.EpisodicStore", DummyStore)
    monkeypatch.setattr("scripts.train_lora.KnowledgeGraph", DummyKG)
    monkeypatch.setattr("scripts.train_lora.PlaceGraph", DummyMap)
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)

    captured: dict[str, int] = {}

    def capture_epi(cfg):
        captured["episodic"] = cfg.num_kv_heads
        return SimpleNamespace()

    def capture_spat(cfg):
        captured["spatial"] = cfg.num_kv_heads
        return SimpleNamespace()

    monkeypatch.setattr("scripts.train_lora.EpisodicAdapter", capture_epi)
    monkeypatch.setattr("scripts.train_lora.SpatialAdapter", capture_spat)
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters",
        lambda *a, **k: {"target_block": 0, "num_blocks": 1},
    )

    cfg = parse_args(
        [
            "dry_run=true",
            "episodic.enabled=true",
            "spatial.enabled=true",
            "replay.enabled=false",
            "efficiency.mqa_gqa=gqa",
            "episodic.num_heads=4",
            "spatial.num_heads=4",
        ]
    )
    train(cfg)
    assert captured == {"episodic": 2, "spatial": 2}

    captured.clear()
    cfg = parse_args(
        [
            "dry_run=true",
            "episodic.enabled=true",
            "spatial.enabled=true",
            "replay.enabled=false",
            "efficiency.mqa_gqa=mqa",
            "episodic.num_heads=4",
            "spatial.num_heads=4",
        ]
    )
    train(cfg)
    assert captured == {"episodic": 1, "spatial": 1}


@pytest.mark.slow
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


@pytest.mark.parametrize("field", ["tokenizer", "processing_class"])
def test_init_sft_trainer_forwards_tokenizer(monkeypatch, field) -> None:
    """Tokenizer or processing class is forwarded to ``SFTTrainer``."""

    captured: dict[str, object] = {}

    namespace: dict[str, object] = {"captured": captured}
    exec(
        f"""
class DummyTrainer:
    def __init__(self, *, model=None, train_dataset=None, args=None, {field}=None, peft_config=None):
        captured.update({{
            'model': model,
            'train_dataset': train_dataset,
            'args': args,
            '{field}': {field},
            'peft_config': peft_config,
        }})
""",
        namespace,
    )

    DummyTrainer = namespace["DummyTrainer"]
    monkeypatch.setattr("scripts.train_lora.SFTTrainer", DummyTrainer)

    _init_sft_trainer("m", "d", "a", "tok", "p")

    assert captured[field] == "tok"


def test_train_skips_retrieval_when_disabled(monkeypatch) -> None:
    """No retrieval occurs when all memory systems are disabled."""

    captured: dict[str, object] = {}

    def fake_loader(_cfg: TrainConfig):
        class DummyModel:
            config = SimpleNamespace(use_cache=False, hidden_size=8)

            def gradient_checkpointing_enable(self) -> None:
                pass

            def __call__(self, input_ids, labels=None, memory_tokens=None):
                captured["memory"] = memory_tokens
                return SimpleNamespace()

        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        return DummyModel(), tokenizer

    def raise_call(*_a, **_k):  # pragma: no cover - should not run
        raise AssertionError("retrieval should be disabled")

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr(
        "scripts.train_lora.EpisodicStore",
        lambda *a, **k: SimpleNamespace(start_background_tasks=lambda *_: None),
    )
    monkeypatch.setattr(
        "scripts.train_lora.KnowledgeGraph",
        lambda *a, **k: SimpleNamespace(start_background_tasks=lambda *_: None),
    )
    monkeypatch.setattr(
        "scripts.train_lora.PlaceGraph",
        lambda *a, **k: SimpleNamespace(start_background_tasks=lambda *_: None),
    )
    monkeypatch.setattr("scripts.train_lora.episodic_retrieve_and_pack", raise_call)
    monkeypatch.setattr("scripts.train_lora.relational_retrieve_and_pack", raise_call)
    monkeypatch.setattr("scripts.train_lora.spatial_retrieve_and_pack", raise_call)
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)

    cfg = parse_args(["dry_run=true", "replay.enabled=false"])
    train(cfg)
    assert captured.get("memory") is None


def test_train_logs_episodic_retrieval(monkeypatch, caplog) -> None:
    """Enabling episodic retrieval logs stats and passes memory to the model."""

    caplog.set_level(logging.INFO)
    captured: dict[str, object] = {}

    def fake_loader(_cfg: TrainConfig):
        class DummyModel:
            config = SimpleNamespace(use_cache=False, hidden_size=8)

            def gradient_checkpointing_enable(self) -> None:
                pass

            def __call__(self, input_ids, labels=None, memory_tokens=None):
                captured["memory"] = memory_tokens
                return SimpleNamespace()

        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        return DummyModel(), tokenizer

    def fake_retrieve(hidden, spec, _store, _proj):
        tokens = torch.zeros(hidden.size(0), spec.k, hidden.size(-1))
        mask = torch.ones(hidden.size(0), spec.k, dtype=torch.bool)
        meta = {"latency_ms": 1.0, "hit_rate": 1.0}
        return MemoryTokens(tokens=tokens, mask=mask, meta=meta)

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr(
        "scripts.train_lora.EpisodicStore",
        lambda *a, **k: SimpleNamespace(start_background_tasks=lambda *_: None),
    )
    monkeypatch.setattr(
        "scripts.train_lora.KnowledgeGraph",
        lambda *a, **k: SimpleNamespace(start_background_tasks=lambda *_: None),
    )
    monkeypatch.setattr(
        "scripts.train_lora.PlaceGraph",
        lambda *a, **k: SimpleNamespace(start_background_tasks=lambda *_: None),
    )
    monkeypatch.setattr("scripts.train_lora.episodic_retrieve_and_pack", fake_retrieve)
    monkeypatch.setattr(
        "scripts.train_lora.relational_retrieve_and_pack",
        lambda *a, **k: MemoryTokens(
            tokens=torch.zeros(0, 0, 0), mask=torch.zeros(0, 0, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr(
        "scripts.train_lora.spatial_retrieve_and_pack",
        lambda *a, **k: MemoryTokens(
            tokens=torch.zeros(0, 0, 0), mask=torch.zeros(0, 0, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)

    cfg = parse_args(
        [
            "dry_run=true",
            "episodic.enabled=true",
            "memory.episodic.enabled=true",
            "memory.episodic.k=2",
            "replay.enabled=false",
        ]
    )
    train(cfg)

    assert "episodic_retrieval_k=2" in caplog.text
    mem = captured.get("memory")
    assert isinstance(mem, MemoryTokens)
    assert mem.tokens.shape[1] == 2


def test_train_relational_retrieval(monkeypatch) -> None:
    """Relational retrieval supplies non-zero tokens to the model."""

    captured: dict[str, object] = {}

    def fake_loader(_cfg: TrainConfig):
        class DummyModel:
            config = SimpleNamespace(use_cache=False, hidden_size=4)

            def gradient_checkpointing_enable(self) -> None:  # pragma: no cover - stub
                pass

            def __call__(self, input_ids, labels=None, memory_tokens=None):
                captured["memory"] = memory_tokens
                return SimpleNamespace()

            def get_input_embeddings(self):  # pragma: no cover - simple embed
                return lambda ids: torch.zeros(ids.shape[0], ids.shape[1], 4)

        tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        return DummyModel(), tokenizer

    def fake_retrieve(hidden, spec, kg, proj):
        tokens = torch.ones(hidden.size(0), spec.k, hidden.size(-1))
        mask = torch.ones(hidden.size(0), spec.k, dtype=torch.bool)
        meta = {"latency_ms": 1.0, "hit_rate": 1.0}
        return MemoryTokens(tokens=tokens, mask=mask, meta=meta)

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr(
        "scripts.train_lora.EpisodicStore",
        lambda *a, **k: SimpleNamespace(start_background_tasks=lambda *_: None),
    )
    monkeypatch.setattr(
        "scripts.train_lora.KnowledgeGraph",
        lambda *a, **k: SimpleNamespace(
            start_background_tasks=lambda *_: None,
            dim=4,
            retrieve=lambda *a, **k: nx.MultiDiGraph(),
        ),
    )
    monkeypatch.setattr(
        "scripts.train_lora.PlaceGraph",
        lambda *a, **k: SimpleNamespace(start_background_tasks=lambda *_: None),
    )
    monkeypatch.setattr("scripts.train_lora.relational_retrieve_and_pack", fake_retrieve)
    monkeypatch.setattr(
        "scripts.train_lora.episodic_retrieve_and_pack",
        lambda *a, **k: MemoryTokens(
            tokens=torch.zeros(0, 0, 0), mask=torch.zeros(0, 0, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr(
        "scripts.train_lora.spatial_retrieve_and_pack",
        lambda *a, **k: MemoryTokens(
            tokens=torch.zeros(0, 0, 0), mask=torch.zeros(0, 0, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)

    cfg = parse_args(
        [
            "dry_run=true",
            "relational=true",
            "memory.relational.enabled=true",
            "memory.relational.k=1",
            "replay.enabled=false",
        ]
    )
    train(cfg)
    mem = captured.get("memory")
    assert isinstance(mem, MemoryTokens)
    assert mem.tokens.abs().sum() > 0
    adapter = RelationalMemoryAdapter()
    hidden = torch.zeros(1, 1, mem.tokens.size(-1))
    out = adapter(hidden, memory=mem)
    assert out.abs().sum() > 0


def _run_training(monkeypatch, overrides: list[str]) -> dict[str, object]:
    """Run training with overrides and return captured writer."""

    writer_holder: dict[str, object] = {}

    class DummyWriter:
        def __init__(self, _store, maxsize: int = 64) -> None:  # pragma: no cover - init
            self.stats = {"writes_enqueued": 0, "writes_committed": 0}
            self.queue = SimpleNamespace(qsize=lambda: 0)
            writer_holder["writer"] = self

        def enqueue(self, key, value) -> None:  # pragma: no cover - simple counter
            self.stats["writes_enqueued"] += 1
            self.queue = SimpleNamespace(qsize=lambda: self.stats["writes_enqueued"])

        def stop(self) -> None:  # pragma: no cover - no cleanup needed
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:  # pragma: no cover - simple model
            super().__init__()
            self.config = SimpleNamespace(use_cache=False, hidden_size=4)

        def gradient_checkpointing_enable(self) -> None:  # pragma: no cover - stub
            pass

        def set_attn_implementation(self, *args, **kwargs) -> None:  # pragma: no cover
            pass

        def forward(self, input_ids, labels=None, memory_tokens=None):
            self._hippo_last_hidden = torch.zeros(input_ids.size(0), input_ids.size(1), 4)
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), 5)
            logits[:, -1, 0] = -1.0
            logits[:, -1, 1] = 1.0
            return SimpleNamespace(logits=logits)

        def get_input_embeddings(self):  # pragma: no cover - simple embed
            return lambda ids: torch.zeros(ids.size(0), ids.size(1), 4)

    class FakeTrainer:
        def __init__(self, model, train_ds, args, tokenizer, peft_config, eval_dataset=None):
            self.model = model
            self.compute_loss = lambda model, inputs, return_outputs=False: (
                torch.tensor(0.0),
                model(input_ids=inputs["input_ids"], labels=inputs["labels"]),
            )

        def train(self) -> None:
            inputs = {
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "labels": torch.ones(1, 2, dtype=torch.long),
            }
            self.compute_loss(self.model, inputs, return_outputs=True)

    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")

    monkeypatch.setattr("scripts.train_lora.AsyncStoreWriter", DummyWriter)
    monkeypatch.setattr(
        "scripts.train_lora._load_model_and_tokenizer", lambda cfg: (DummyModel(), tokenizer)
    )
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters", lambda *a, **k: {"target_block": 0, "num_blocks": 1}
    )
    monkeypatch.setattr("scripts.train_lora._init_sft_trainer", FakeTrainer)
    monkeypatch.setattr(
        "scripts.train_lora.prepare_datasets", lambda cfg: ([{"text": "hi"}], [{"text": "hi"}])
    )
    monkeypatch.setattr("scripts.train_lora.count_trainable_parameters", lambda _m: 1)
    monkeypatch.setattr("scripts.train_lora.SFTConfig", lambda **kw: SimpleNamespace(**kw))

    cfg = parse_args(["max_steps=1", "write_threshold=0.0", *overrides])
    train(cfg)
    return writer_holder


def test_write_gate_runs_during_training(monkeypatch) -> None:
    """Gating decisions enqueue writes during normal training."""

    writer_holder = _run_training(monkeypatch, [])
    writer = writer_holder.get("writer")
    assert writer is not None
    assert writer.stats["writes_enqueued"] > 0


def test_disable_writes_flag(monkeypatch) -> None:
    """Disabling writes skips enqueueing despite gate scores."""

    writer_holder = _run_training(monkeypatch, ["memory.runtime.enable_writes=false"])
    writer = writer_holder.get("writer")
    assert writer is not None
    assert writer.stats["writes_enqueued"] == 0


def test_gate_logging(monkeypatch, caplog) -> None:
    """Gate stats are logged at the configured interval."""

    caplog.set_level(logging.INFO)
    _run_training(monkeypatch, ["memory.runtime.log_interval=1"])
    assert any("gate_accepts" in r.message for r in caplog.records)
