import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from hippo_eval import datasets as build_datasets
from hippo_mem.training import jsonl_dataset

SUITES = ["episodic", "semantic", "spatial"]


def test_dataset_generators_deterministic(tmp_path: Path) -> None:
    """Datasets with same seed should produce identical JSONL outputs."""

    for suite in SUITES:
        suite_dir = tmp_path / suite
        suite_dir.mkdir()
        items1 = build_datasets.generate_dataset(suite, size=5, seed=123)
        items2 = build_datasets.generate_dataset(suite, size=5, seed=123)
        assert items1 == items2

        file1 = suite_dir / f"{suite}_a.jsonl"
        file2 = suite_dir / f"{suite}_b.jsonl"
        build_datasets.write_jsonl(file1, items1)
        build_datasets.write_jsonl(file2, items2)

        assert file1.read_text() == file2.read_text()
        assert len(file1.read_text().splitlines()) == 5

        checksum_file = suite_dir / "checksums.json"
        hash_written = build_datasets.record_checksum(file1, checksum_file)
        data = json.loads(checksum_file.read_text())
        assert data[file1.name] == hash_written
        assert hash_written == build_datasets.sha256_file(file1)

        build_datasets.update_dataset_card(suite, suite_dir, file1.name, hash_written, "test")
        card = json.loads((suite_dir / "dataset_card.json").read_text())
        assert card["files"][file1.name] == hash_written
        assert card["suite"] == suite


def test_semantic_options() -> None:
    """Semantic generator supports hop depth and contradictions."""

    three_hop = build_datasets.generate_semantic(1, seed=0, hop_depth=3)
    assert "was sold at" in three_hop[0]["prompt"]

    contradict = build_datasets.generate_semantic(1, seed=0, inject_contradictions=True)
    assert "However, others report" in contradict[0]["prompt"]


def test_semantic_require_memory_omits_facts() -> None:
    """When ``require_memory`` is set, prompts exclude fact sentences."""

    item = build_datasets.generate_semantic(1, seed=0, require_memory=True, profile="hard")[0]
    for fact in item["facts"]:
        assert fact["text"] not in item["prompt"]


def test_semantic_paraphrasing_and_pronouns() -> None:
    """Paraphrasing and pronoun ambiguity can be enabled."""

    item = build_datasets.generate_semantic(
        1,
        seed=0,
        hop_depth=3,
        paraphrase_prob=1.0,
        ambiguity_prob=1.0,
    )[0]
    prompt = item["prompt"]
    assert "It" in prompt
    assert "purchased" in prompt or "could be found at" in prompt or "is located in" in prompt


def test_semantic_three_hop_facts() -> None:
    """3-hop chains emit three schema-fit facts in order."""

    item = build_datasets.generate_semantic(1, seed=0, hop_depth=3, require_memory=True)[0]
    facts = [f["text"] for f in item["facts"] if f["schema_fit"]]
    assert any("bought" in f or "purchased" in f for f in facts)
    assert any("sold" in f or "found" in f for f in facts)
    assert any("is in" in f or "is located" in f for f in facts)


def test_episodic_flags() -> None:
    """Episodic items expose reward and pin flags."""

    item = build_datasets.generate_episodic(1, seed=0)[0]
    assert "reward" in item and "pin" in item


def test_episodic_single_protagonist_event() -> None:
    """Prompts mention the queried protagonist exactly once before the question."""

    items = build_datasets.generate_episodic(50, seed=0)
    people = ["Alice", "Bob", "Carol", "Dave"]
    for item in items:
        prompt = item["prompt"]
        for name in people:
            if (
                f"What did {name} do?" in prompt
                or f"Where was {name}?" in prompt
                or f"When was {name} at" in prompt
            ):
                assert prompt.count(name) == 2


def test_semantic_fact_labels() -> None:
    """Semantic items include per-fact schema-fit metadata."""

    item = build_datasets.generate_semantic(1, seed=0, hop_depth=3, inject_contradictions=True)[0]
    facts = item["facts"]
    assert facts and all("schema_fit" in f and "time" in f for f in facts)


def test_spatial_moves() -> None:
    """Spatial generator emits canonical move string answers."""

    items = build_datasets.generate_spatial(4, seed=0)
    assert all(set(item["answer"]) <= {"U", "D", "L", "R"} for item in items)


def test_loader_rejects_corrupted_json(tmp_path: Path) -> None:
    """Corrupted JSON lines raise :class:`JSONDecodeError`."""

    file = tmp_path / "bad.jsonl"
    file.write_text('{"prompt": "p1", "answer": "a1"}\n{bad line}\n')
    with pytest.raises(json.JSONDecodeError) as exc:
        jsonl_dataset.load_jsonl_files([str(file)])
    assert "Expecting" in str(exc.value)


def _collect_batches(ds, seed: int) -> list[list[str]]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=2, shuffle=True, generator=generator)
    return [batch["text"] for batch in loader]


def test_data_loader_seed_determinism(tmp_path: Path) -> None:
    """DataLoader shuffling is deterministic for a given seed."""

    file = tmp_path / "data.jsonl"
    items = [{"prompt": f"p{i}", "answer": f"a{i}"} for i in range(6)]
    with file.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    ds = jsonl_dataset.load_jsonl_files([str(file)])
    run1 = _collect_batches(ds, 1337)
    run2 = _collect_batches(ds, 1337)
    run3 = _collect_batches(ds, 4242)
    assert run1 == run2
    assert run1 != run3
