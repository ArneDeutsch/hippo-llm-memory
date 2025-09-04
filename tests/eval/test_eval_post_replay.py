import csv
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_post_replay_cycle_generates_metrics(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/eval_bench.py",
        "suite=episodic",
        "preset=memory/hei_nw",
        "n=1",
        "seed=0",
        "post_replay_cycles=1",
        f"outdir={outdir}",
    ]
    subprocess.run(cmd, check=True)
    csv_path = outdir / "metrics.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        expected_headers = {
            "idx",
            "prompt",
            "answer",
            "pred",
            "em_raw",
            "em_norm",
            "f1",
            "pred_len",
            "gold_len",
            "overlong",
            "format_violation",
            "latency_ms",
            "time_ms_per_100",
            "rss_mb",
            "flags",
            "gating_enabled",
        }
        assert expected_headers.issubset(headers)
        rows = list(reader)

    flags = {row["flags"] for row in rows}
    assert "pre_replay" in flags
    assert any(flag.startswith("post_replay") for flag in flags)

    pre = {row["prompt"]: row for row in rows if row["flags"] == "pre_replay"}
    post = {row["prompt"]: row for row in rows if row["flags"].startswith("post_replay")}
    assert pre and post
    for prompt, pre_row in pre.items():
        post_row = post[prompt]
        assert float(pre_row["latency_ms"]) != float(post_row["latency_ms"])
