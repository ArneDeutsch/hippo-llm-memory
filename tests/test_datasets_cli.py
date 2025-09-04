import json
import subprocess
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_cli_generates_episodic_multi(tmp_path: Path) -> None:
    out = tmp_path / "epi_multi.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "hippo_eval.datasets.cli",
        "--suite",
        "episodic_multi",
        "--size",
        "10",
        "--seed",
        "0",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    data = _read_jsonl(out)
    assert len(data) == 10
    assert any("Actually" in item["prompt"] for item in data)


def test_cli_episodic_cross_capacity(tmp_path: Path) -> None:
    out = tmp_path / "epi_cross.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "hippo_eval.datasets.cli",
        "--suite",
        "episodic_cross",
        "--size",
        "1000",
        "--seed",
        "0",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    data = _read_jsonl(out)
    assert len(data) == 1000
