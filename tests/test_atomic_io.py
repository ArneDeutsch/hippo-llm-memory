import json
import threading

from hippo_mem.common.io import atomic_write_jsonl, read_jsonl


def test_atomic_write_jsonl_thread_safe(tmp_path):
    file = tmp_path / "data.jsonl"
    records_a = [{"id": 1}, {"id": 2}]
    records_b = [{"id": 3}]

    def write(records):
        atomic_write_jsonl(file, records)

    t1 = threading.Thread(target=write, args=(records_a,))
    t2 = threading.Thread(target=write, args=(records_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert file.exists()

    data = list(read_jsonl(file))
    assert data in (records_a, records_b)

    expected_size = sum(len(json.dumps(r)) + 1 for r in data)
    assert file.stat().st_size == expected_size

    with file.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == len(data)
    assert all(line.endswith("\n") for line in lines)
