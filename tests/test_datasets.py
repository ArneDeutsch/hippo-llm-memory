import json
from pathlib import Path

from hippo_mem.eval import datasets as build_datasets

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


def test_episodic_flags() -> None:
    """Episodic items expose reward and pin flags."""

    item = build_datasets.generate_episodic(1, seed=0)[0]
    assert "reward" in item and "pin" in item


def test_semantic_fact_labels() -> None:
    """Semantic items include per-fact schema-fit metadata."""

    item = build_datasets.generate_semantic(1, seed=0, hop_depth=3, inject_contradictions=True)[0]
    facts = item["facts"]
    assert facts and all("schema_fit" in f and "time" in f for f in facts)


def test_spatial_trajectory() -> None:
    """Spatial generator emits random walk trajectories for path integration."""

    items = build_datasets.generate_spatial(3, seed=0)
    traj_items = [t for t in items if "trajectory" in t]
    assert traj_items and traj_items[0]["trajectory"][-1] == traj_items[0]["answer"]
