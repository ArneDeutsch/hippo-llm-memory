import json
import subprocess
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_cli_generates_episodic(tmp_path: Path) -> None:
    out_dir = tmp_path / "episodic_cross_mem"
    cmd = [
        sys.executable,
        "-m",
        "hippo_eval.datasets.cli",
        "--suite",
        "episodic_cross_mem",
        "--size",
        "10",
        "--seed",
        "0",
        "--out",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    teach = _read_jsonl(out_dir / "episodic_cross_mem_teach.jsonl")
    test = _read_jsonl(out_dir / "episodic_cross_mem_test.jsonl")
    assert len(test) == 10
    assert teach and test


def test_cli_require_memory_semantic(tmp_path: Path) -> None:
    out_dir = tmp_path / "semantic_mem"
    cmd = [
        sys.executable,
        "-m",
        "hippo_eval.datasets.cli",
        "--suite",
        "semantic_mem",
        "--size",
        "5",
        "--seed",
        "0",
        "--out",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    teach = _read_jsonl(out_dir / "semantic_mem_teach.jsonl")
    test = _read_jsonl(out_dir / "semantic_mem_test.jsonl")
    assert teach and test
    keys = {t["context_key"] for t in teach}
    assert keys == {t["context_key"] for t in test}
