import json
import sys
import threading
import types
from pathlib import Path

from hippo_mem.common.io import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_parquet,
)


def test_atomic_write_json_thread_safe(tmp_path):
    file = tmp_path / "data.json"
    obj_a = {"a": 1}
    obj_b = {"b": 2}

    def write(obj):
        atomic_write_json(file, obj)

    t1 = threading.Thread(target=write, args=(obj_a,))
    t2 = threading.Thread(target=write, args=(obj_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert read_json(file) in (obj_a, obj_b)


def test_atomic_write_jsonl_no_interleaving(tmp_path):
    file = tmp_path / "data.jsonl"
    records_a = [{"id": 1}, {"id": 2}]
    records_b = [{"id": 3}, {"id": 4}]

    def write(records):
        atomic_write_jsonl(file, records)

    t1 = threading.Thread(target=write, args=(records_a,))
    t2 = threading.Thread(target=write, args=(records_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # File must exist after concurrent writes
    assert file.exists()

    # File size should match one of the complete record sets
    size_a = sum(len(json.dumps(rec)) + 1 for rec in records_a)
    size_b = sum(len(json.dumps(rec)) + 1 for rec in records_b)
    assert file.stat().st_size in (size_a, size_b)

    # Reload the file and ensure no partial lines are present
    content = file.read_text()
    assert content.endswith("\n")
    lines = content.splitlines()
    assert len(lines) in (len(records_a), len(records_b))

    # Only one complete record set should exist
    parsed = [json.loads(line) for line in lines]
    assert parsed in (records_a, records_b)


def test_read_parquet_mock(monkeypatch, tmp_path):
    file = tmp_path / "data.parquet"
    file.touch()

    calls: list[Path] = []

    def fake_read_parquet(path):
        calls.append(path)
        return "df"

    dummy = types.SimpleNamespace(read_parquet=fake_read_parquet)
    monkeypatch.setitem(sys.modules, "pandas", dummy)

    assert read_parquet(file) == "df"
    assert calls == [file]
