from pathlib import Path

from scripts import build_datasets

SUITES = ["episodic", "semantic", "spatial"]


def test_dataset_generators_deterministic(tmp_path: Path) -> None:
    """Datasets with same seed should produce identical JSONL outputs."""

    for suite in SUITES:
        items1 = build_datasets.generate_dataset(suite, size=5, seed=123)
        items2 = build_datasets.generate_dataset(suite, size=5, seed=123)
        assert items1 == items2

        file1 = tmp_path / f"{suite}_a.jsonl"
        file2 = tmp_path / f"{suite}_b.jsonl"
        build_datasets.write_jsonl(file1, items1)
        build_datasets.write_jsonl(file2, items2)

        assert file1.read_text() == file2.read_text()
        assert len(file1.read_text().splitlines()) == 5

        checksum_file = tmp_path / "checksums.txt"
        hash_written = build_datasets.record_checksum(file1, checksum_file)
        last_line = checksum_file.read_text().strip().splitlines()[-1]
        assert last_line == f"{hash_written}  {file1.name}"
        assert hash_written == build_datasets.sha256_file(file1)


def test_semantic_hops_and_contradictions() -> None:
    """Semantic generator supports hop depth and contradiction injection."""

    items1 = build_datasets.generate_semantic(size=1, seed=42, hops=3, contradictions=True)
    items2 = build_datasets.generate_semantic(size=1, seed=42, hops=3, contradictions=True)
    assert items1 == items2

    prompt = items1[0]["prompt"]
    # Two location statements: one true and one contradictory
    assert prompt.count("is in") == 2
    assert "However" in prompt and "was sold at" in prompt
    assert items1[0]["answer"] in prompt
