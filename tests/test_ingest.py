import json
from types import SimpleNamespace

import networkx as nx

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

    created: dict[str, object] = {}

    class DummyKG:
        def __init__(self, config=None) -> None:
            self.graph = nx.MultiDiGraph()
            created["kg"] = self

        def start_background_tasks(self, _interval) -> None:
            pass

        def ingest(self, tup):
            h, r, t, c, *_ = tup
            self.graph.add_edge(h, t, relation=r, context=c)
            return True

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
    monkeypatch.setattr("scripts.train_lora.AsyncStoreWriter", DummyWriter)
    monkeypatch.setattr("scripts.train_lora.KnowledgeGraph", DummyKG)
    monkeypatch.setattr("scripts.train_lora.PlaceGraph", DummyMap)
    monkeypatch.setattr(
        "scripts.train_lora.attach_adapters", lambda *a, **k: {"target_block": 0, "num_blocks": 1}
    )
    monkeypatch.setattr("scripts.train_lora.log_memory_status", lambda *a, **k: None)

    cfg = TrainConfig(
        dry_run=True,
        fast_track_ingest=True,
        train_files=[str(data_file)],
        val_files=[str(data_file)],
    )
    train(cfg)
    kg = created["kg"]
    assert kg.graph.number_of_nodes() > 0
    assert kg.graph.number_of_edges() > 0
