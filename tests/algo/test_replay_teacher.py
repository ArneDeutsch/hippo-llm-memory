# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import json

from hippo_mem.consolidation.replay_dataset import ReplayDataset


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_teacher_outputs_added_once(tmp_path):
    session = "s"
    base = tmp_path / "hei_nw" / session
    base.mkdir(parents=True)

    _write_jsonl(
        base / "episodic.jsonl",
        [{"schema": "episodic.v1", "id": 1, "key": [0.0], "value": {}}],
    )
    _write_jsonl(base / "relational.jsonl", [])
    _write_jsonl(base / "spatial.jsonl", [])

    calls = {"n": 0}

    def teacher(rec):
        calls["n"] += 1
        return {"text": f"T{rec['id']}", "logits": [1.0]}

    ds = ReplayDataset(
        str(tmp_path / "hei_nw"),
        session,
        ratios={"episodic": 1.0},
        seed=0,
        teacher=teacher,
    )

    it = iter(ds)
    first = next(it)
    assert first["teacher"]["text"] == "T1"
    assert calls["n"] == 1

    second = next(it)
    assert second["teacher"]["text"] == "T1"
    assert calls["n"] == 1
