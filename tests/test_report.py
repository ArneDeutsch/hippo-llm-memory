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
    base_dir = tmp_path / "runs" / "20250101" / "baselines" / "core"
    runs_on = base_dir / "gate_on" / "episodic"
    runs_off = base_dir / "gate_off" / "episodic"
    gates_on = {
        "relational": {
            "attempts": 10,
            "inserted": 8,
            "aggregated": 2,
            "routed_to_episodic": 0,
        },
        "spatial": {
            "attempts": 20,
            "inserted": 19,
            "aggregated": 1,
            "blocked_new_edges": 1,
            "nodes_added": 5,
            "edges_added": 10,
        },
    }
    gates_off = {
        "relational": {
            "attempts": 10,
            "inserted": 10,
            "aggregated": 0,
            "routed_to_episodic": 0,
        },
        "spatial": {
            "attempts": 20,
            "inserted": 20,
            "aggregated": 0,
            "blocked_new_edges": 0,
            "nodes_added": 10,
            "edges_added": 20,
        },
    }
    _make_metrics(
        runs_on / "50_1337",
        "episodic",
        {"em": 0.5, "r": 0.7},
        {"tokens": 10},
        gates_on,
    )
    _make_metrics(
        runs_on / "200_2025",
        "episodic",
        {"em": 0.7, "r": 0.9},
        {"tokens": 30},
        gates_on,
    )
    _make_metrics(
        runs_off / "50_4242",
        "episodic",
        {"em": 0.6, "r": 0.8},
        {"tokens": 20},
        gates_off,
    )

    base = tmp_path / "runs" / "20250101"
    metrics = collect_metrics(base)
    summary = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))

    assert summary["episodic"]["baselines/core/gate_on"]["em"] == 0.6
    assert summary["episodic"]["baselines/core/gate_on"]["tokens"] == 20
    out = tmp_path / "reports" / "20250101"
    paths = write_reports(summary, retrieval, gates, out, plots=False)
    md_path = paths["episodic"]
    assert md_path.exists()
    text = md_path.read_text()
    assert "duplicate_rate" in text
    assert "nodes_per_1k" in text
    assert "Gate ON vs OFF" in text
    assert "+0.200" in text
    assert "-250.000" in text
