import json
from types import SimpleNamespace

import torch

from hippo_mem.common import MemoryTokens
from hippo_mem.relational.kg import KnowledgeGraph
from scripts.train_lora import TrainConfig, train


def test_fast_track_ingestion(monkeypatch, tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(json.dumps({"prompt": "Alice likes Bob.", "answer": ""}) + "\n")

    def fake_loader(_cfg):
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

    class DummyWriter:
        def __init__(self, _store) -> None:
            self.stats = {"writes_enqueued": 0, "writes_committed": 0}

        def enqueue(self, *_args, **_kwargs) -> None:
            pass

        def stop(self) -> None:
            pass

    kg = KnowledgeGraph(config={"schema_threshold": 0.5})
    kg.schema_index.add_schema("likes", "likes")
    kg.start_background_tasks = lambda _interval: None

    class DummyMap:
        def __init__(self, path_integration=False, config=None) -> None:
            pass

        def start_background_tasks(self, _interval) -> None:
            pass

        def log_status(self) -> str:
            return "map"

    def empty_mem(*_a, **_k):
        return MemoryTokens(tokens=torch.zeros(0, 0, 0), mask=torch.zeros(0, 0, dtype=torch.bool))

    monkeypatch.setattr("scripts.train_lora._load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr("scripts.train_lora.EpisodicStore", DummyStore)
    monkeypatch.setattr("scripts.train_lora.AsyncWriter", DummyWriter)
    monkeypatch.setattr("scripts.train_lora.KnowledgeGraph", lambda config=None: kg)
    monkeypatch.setattr("scripts.train_lora.PlaceGraph", DummyMap)
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters", lambda *a, **k: {"target_block": 0, "num_blocks": 1}
    )
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)
    monkeypatch.setattr("scripts.train_lora.episodic_retrieve_and_pack", empty_mem)
    monkeypatch.setattr("scripts.train_lora.relational_retrieve_and_pack", empty_mem)
    monkeypatch.setattr("scripts.train_lora.spatial_retrieve_and_pack", empty_mem)

    cfg = TrainConfig(
        dry_run=True,
        schema_fasttrack_ingest=True,
        train_files=[str(data_file)],
        val_files=[str(data_file)],
    )
    train(cfg)

    assert kg.graph.number_of_nodes() > 0
    assert kg.graph.number_of_edges() > 0
