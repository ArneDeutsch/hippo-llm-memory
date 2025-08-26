import itertools
import json

from hippo_mem.consolidation.replay_dataset import ReplayDataset


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_uniform_policy(tmp_path):
    base = tmp_path / "s"
    base.mkdir()
    _write_jsonl(
        base / "episodic.jsonl",
        [
            {"schema": "episodic.v1", "id": 1, "key": [0.0], "value": {}, "salience": 0.01},
            {"schema": "episodic.v1", "id": 2, "key": [0.0], "value": {}, "salience": 1.0},
        ],
    )
    _write_jsonl(base / "relational.jsonl", [])
    _write_jsonl(base / "spatial.jsonl", [])

    ds = ReplayDataset(str(tmp_path), "s", ratios={"episodic": 1.0}, seed=0, policy="uniform")
    samples = list(itertools.islice(iter(ds), 500))
    counts = {1: 0, 2: 0}
    for rec in samples:
        counts[rec["id"]] += 1
    total = sum(counts.values())
    assert abs(counts[1] / total - 0.5) <= 0.1
    assert abs(counts[2] / total - 0.5) <= 0.1


def test_cycles_and_max_items(tmp_path):
    base = tmp_path / "s"
    base.mkdir()
    _write_jsonl(
        base / "episodic.jsonl",
        [
            {"schema": "episodic.v1", "id": 1, "key": [0.0], "value": {}},
            {"schema": "episodic.v1", "id": 2, "key": [0.0], "value": {}},
        ],
    )
    _write_jsonl(base / "relational.jsonl", [])
    _write_jsonl(base / "spatial.jsonl", [])

    ds = ReplayDataset(
        str(tmp_path), "s", ratios={"episodic": 1.0}, seed=0, policy="spaced", cycles=2
    )
    items = list(ds)
    ids = [rec["id"] for rec in items]
    assert len(items) == 4
    assert ids.count(1) == 2 and ids.count(2) == 2

    ds2 = ReplayDataset(str(tmp_path), "s", ratios={"episodic": 1.0}, seed=0, max_items=3)
    items2 = list(ds2)
    assert len(items2) == 3


def test_noise_level_enables_zero_weight_items(tmp_path):
    base = tmp_path / "s"
    base.mkdir()
    _write_jsonl(
        base / "episodic.jsonl",
        [
            {"schema": "episodic.v1", "id": 1, "key": [0.0], "value": {}, "salience": 1.0},
            {"schema": "episodic.v1", "id": 2, "key": [0.0], "value": {}, "salience": 0.0},
        ],
    )
    _write_jsonl(base / "relational.jsonl", [])
    _write_jsonl(base / "spatial.jsonl", [])

    ds_no_noise = ReplayDataset(
        str(tmp_path), "s", ratios={"episodic": 1.0}, seed=0, noise_level=0.0
    )
    samples = list(itertools.islice(iter(ds_no_noise), 50))
    assert all(rec["id"] == 1 for rec in samples)

    ds_noise = ReplayDataset(str(tmp_path), "s", ratios={"episodic": 1.0}, seed=0, noise_level=1.0)
    noisy_samples = list(itertools.islice(iter(ds_noise), 50))
    assert any(rec["id"] == 2 for rec in noisy_samples)
