from __future__ import annotations

import csv
import json
from pathlib import Path

from hippo_eval.harness.io import write_csv, write_meta, write_metrics


def test_write_read_roundtrip(tmp_path: Path) -> None:
    meta = {"suite": "episodic_cross_mem", "preset": "baselines/core", "n": 2}
    metrics = {"metrics": {"episodic_cross_mem": {"pre_em": 1.0}}}
    rows = [{"a": 1, "b": 2}]
    write_meta(tmp_path / "meta.json", meta)
    write_metrics(tmp_path / "metrics.json", metrics)
    write_csv(tmp_path / "rows.csv", rows)
    assert json.loads((tmp_path / "meta.json").read_text())["suite"] == "episodic_cross_mem"
    assert (
        json.loads((tmp_path / "metrics.json").read_text())["metrics"]["episodic_cross_mem"][
            "pre_em"
        ]
        == 1.0
    )
    with (tmp_path / "rows.csv").open() as fh:
        assert list(csv.DictReader(fh))[0]["a"] == "1"
