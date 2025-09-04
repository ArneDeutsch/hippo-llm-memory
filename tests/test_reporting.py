from __future__ import annotations

import json
from pathlib import Path

import pytest

from hippo_eval.reporting.report import write_reports
from hippo_eval.reporting.rollup import collect_metrics, summarise


def _make_metrics(path: Path, suite: str, metrics: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    content = {"metrics": {suite: metrics}}
    (path / "metrics.json").write_text(json.dumps(content))


def test_missing_pre_flagged(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic"
    _make_metrics(base / "50_1337", "episodic", {"post_em": 0.5})

    root = tmp_path / "runs" / "20250101"
    summary, _ = summarise(collect_metrics(root))
    out_dir = tmp_path / "reports" / "20250101"
    write_reports(summary, {}, {}, {}, out_dir, plots=False, seed_count=1)
    text = (out_dir / "episodic" / "summary.md").read_text()
    assert "_missing_" in text
    assert "MissingPre" in text


def test_average_ignores_missing(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic"
    _make_metrics(base / "50_1337", "episodic", {"pre_em": 0.4, "post_em": 0.6})
    _make_metrics(base / "50_2025", "episodic", {"post_em": 0.8})

    summary, _ = summarise(collect_metrics(tmp_path / "runs" / "20250101"))
    metrics = summary["episodic"]["baselines/core"][50]
    assert metrics["pre_em"][0] == 0.4
    assert metrics["post_em"][0] == pytest.approx(0.7)


def test_collect_metrics_skips_none(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic" / "50_1337"
    base.mkdir(parents=True, exist_ok=True)
    content = {
        "metrics": {"episodic": {"em": 0.5}},
        "diagnostics": {"episodic": {"foo": None}},
        "n": 5,
    }
    (base / "metrics.json").write_text(json.dumps(content))
    data = collect_metrics(tmp_path / "runs" / "20250101")
    record = data[("episodic", "baselines/core", 50)][0]
    assert "foo" not in record
