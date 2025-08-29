import itertools
import json

from hippo_mem.consolidation.replay_dataset import ReplayDataset


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_replay_dataset_ratios_and_priority(tmp_path):
    session = "s"
    base = tmp_path / "hei_nw" / session
    base.mkdir(parents=True)

    # episodic store with different salience
    _write_jsonl(
        base / "episodic.jsonl",
        [
            {
                "schema": "episodic.v1",
                "id": 1,
                "key": [0.0],
                "value": {},
                "ts": 0.0,
                "salience": 0.1,
            },
            {
                "schema": "episodic.v1",
                "id": 2,
                "key": [0.0],
                "value": {},
                "ts": 1.0,
                "salience": 1.0,
            },
        ],
    )

    _write_jsonl(
        base / "relational.jsonl",
        [
            {
                "schema": "relational.v1",
                "type": "edge",
                "src": "a",
                "relation": "r",
                "dst": "b",
            }
        ],
    )

    _write_jsonl(
        base / "spatial.jsonl",
        [
            {"schema": "spatial.v1", "type": "meta"},
            {
                "schema": "spatial.v1",
                "type": "edge",
                "src": 1,
                "dst": 2,
                "cost": 1.0,
                "success": 1.0,
                "last_seen": 0,
                "weight": 1.0,
            },
        ],
    )

    ds = ReplayDataset(
        str(tmp_path / "hei_nw"),
        session,
        ratios={"episodic": 0.6, "relational": 0.3, "spatial": 0.1},
        seed=0,
    )

    samples = list(itertools.islice(iter(ds), 1000))
    counts = {"episodic": 0, "relational": 0, "spatial": 0}
    epi_ids = []
    for rec in samples:
        counts[rec["kind"]] += 1
        if rec["kind"] == "episodic":
            epi_ids.append(rec["id"])

    total = len(samples)
    assert total == 1000
    assert abs(counts["episodic"] / total - 0.6) <= 0.05
    assert abs(counts["relational"] / total - 0.3) <= 0.05
    assert abs(counts["spatial"] / total - 0.1) <= 0.05

    # higher-salience episodic id=2 should be sampled more often
    assert epi_ids.count(2) > epi_ids.count(1)
