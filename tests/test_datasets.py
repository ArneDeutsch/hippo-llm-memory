from pathlib import Path

from scripts import build_datasets

SUITES = ["episodic", "semantic", "spatial"]


def test_dataset_generators_deterministic(tmp_path: Path) -> None:
    """Datasets with same seed should produce identical JSONL outputs."""

    for suite in SUITES:
        items1 = build_datasets.generate_dataset(suite, n=5, seed=123)
        items2 = build_datasets.generate_dataset(suite, n=5, seed=123)
        assert items1 == items2

        file1 = tmp_path / f"{suite}_a.jsonl"
        file2 = tmp_path / f"{suite}_b.jsonl"
        build_datasets.write_jsonl(file1, items1)
        build_datasets.write_jsonl(file2, items2)

        assert file1.read_text() == file2.read_text()
        assert len(file1.read_text().splitlines()) == 5
