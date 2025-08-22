from __future__ import annotations

import json
from pathlib import Path

from scripts.report import (
    collect_gates,
    collect_metrics,
    collect_retrieval,
    summarise,
    summarise_gates,
    summarise_retrieval,
    write_reports,
)


def _make_metrics(
    path: Path, suite: str, metrics: dict, compute: dict | None = None, gates: dict | None = None
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    content = {"metrics": {suite: metrics}}
    if compute:
        content["metrics"]["compute"] = compute
    if gates:
        content["gates"] = gates
    (path / "metrics.json").write_text(json.dumps(content))


def test_report_aggregation(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic"
    gates_sample = {
        "relational": {"attempts": 1, "inserted": 1, "aggregated": 0, "routed_to_episodic": 0},
        "spatial": {"attempts": 2, "inserted": 2, "aggregated": 0, "blocked_new_edges": 1},
    }
    _make_metrics(
        runs / "50_1337",
        "episodic",
        {"em": 0.5, "r": 0.7},
        {"tokens": 10},
        gates_sample,
    )
    _make_metrics(
        runs / "200_2025",
        "episodic",
        {"em": 0.7, "r": 0.9},
        {"tokens": 30},
        gates_sample,
    )

    base = tmp_path / "runs" / "20250101"
    metrics = collect_metrics(base)
    summary = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))

    assert summary["episodic"]["baselines/core"]["em"] == 0.6
    assert summary["episodic"]["baselines/core"]["tokens"] == 20
    out = tmp_path / "reports" / "20250101"
    paths = write_reports(summary, retrieval, gates, out, plots=False)
    md_path = paths["episodic"]
    assert md_path.exists()
    text = md_path.read_text()
    assert "episodic Summary" in text
    assert "Gate Telemetry" in text
