from __future__ import annotations

import json
from pathlib import Path

from scripts.report import collect_metrics, summarise, write_report


def _make_metrics(path: Path, suite: str, metrics: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    content = {suite: metrics}
    (path / "metrics.json").write_text(json.dumps(content))


def test_report_aggregation(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic"
    _make_metrics(runs / "50_1337", "episodic", {"em": 0.5, "r": 0.7})
    _make_metrics(runs / "200_2025", "episodic", {"em": 0.7, "r": 0.9})

    data = collect_metrics(tmp_path / "runs" / "20250101" / "baselines")
    summary = summarise(data)

    assert summary["episodic"]["core"]["em"] == 0.6
    out = tmp_path / "reports" / "20250101"
    md_path = write_report(summary, out, plots=False)
    assert md_path.exists()
    assert "Baseline Summary" in md_path.read_text()
