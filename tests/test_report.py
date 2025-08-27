from __future__ import annotations

import json
from pathlib import Path

from scripts.report import (
    _find_latest_date,
    collect_gates,
    collect_metrics,
    collect_retrieval,
    summarise,
    summarise_gates,
    summarise_retrieval,
    write_reports,
    write_smoke,
)


def _make_metrics(
    path: Path,
    suite: str,
    metrics: dict,
    compute: dict | None = None,
    gates: dict | None = None,
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
        {"em_raw": 0.5, "r": 0.7},
        {"total_tokens": 10, "rss_mb": 1.0, "time_ms_per_100": 2.0},
        gates_on,
    )
    _make_metrics(
        runs_on / "50_2025",
        "episodic",
        {"em_raw": 0.7, "r": 0.9},
        {"total_tokens": 30, "rss_mb": 1.5, "time_ms_per_100": 4.0},
        gates_on,
    )
    _make_metrics(
        runs_off / "50_4242",
        "episodic",
        {"em_raw": 0.6, "r": 0.8},
        {"total_tokens": 20, "rss_mb": 2.0, "time_ms_per_100": 3.0},
        gates_off,
    )
    # also create another suite to ensure per-suite report generation
    other = tmp_path / "runs" / "20250101" / "baselines" / "core" / "semantic"
    _make_metrics(
        other / "50_1337",
        "semantic",
        {"f1": 0.4},
        {"total_tokens": 5, "rss_mb": 0.5, "time_ms_per_100": 1.0},
    )

    base = tmp_path / "runs" / "20250101"
    metrics = collect_metrics(base)
    summary = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))

    em_stats = summary["episodic"]["baselines/core/gate_on"][50]["em_raw"]
    assert em_stats[0] == 0.6
    assert round(em_stats[1], 3) == 0.141

    out = tmp_path / "reports" / "20250101"
    paths = write_reports(summary, retrieval, gates, out, plots=False)
    # per-suite summaries present
    assert set(paths.keys()) == {"episodic", "semantic"}
    md_path = paths["episodic"]
    text = md_path.read_text()
    # table header contains all fields
    assert "| Preset | Size | EM (raw) | r | rss_mb | time_ms_per_100 | total_tokens |" in text
    # both presets appear as rows
    assert "| baselines/core/gate_on | 50 |" in text
    assert "| baselines/core/gate_off | 50 |" in text
    # gate telemetry rendered
    assert "duplicate_rate" in text
    assert "nodes_per_1k" in text
    assert "Gate ON vs OFF" in text
    assert "+0.200" in text
    assert "-250.000" in text

    idx = out / "index.md"
    assert idx.exists()
    idx_text = idx.read_text()
    header = next(line for line in idx_text.splitlines() if line.startswith("| Suite"))
    for col in ["EM (raw)", "r", "rss_mb", "time_ms_per_100", "total_tokens"]:
        assert col in header
    assert "[episodic](episodic/summary.md)" in idx_text
    assert "[semantic](semantic/summary.md)" in idx_text


def test_report_handles_missing_optional(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic"
    _make_metrics(base_dir / "50_1337", "episodic", {"em": 0.5}, {"total_tokens": 10})
    base = tmp_path / "runs" / "20250101"
    summary = summarise(collect_metrics(base))
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))
    out = tmp_path / "reports" / "20250101"
    paths = write_reports(summary, retrieval, gates, out, plots=False)
    text = paths["episodic"].read_text()
    assert "Retrieval Telemetry" not in text
    assert "Gate Telemetry" not in text
    assert (out / "index.md").exists()


def test_find_latest_date(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    (base / "20240101").mkdir(parents=True)
    (base / "20250102").mkdir()
    assert _find_latest_date(base) == "20250102"


def test_smoke_report(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for suite in ["episodic", "semantic", "spatial"]:
        suite_dir = data_root / suite
        suite_dir.mkdir(parents=True)
        (suite_dir / "50_1337.jsonl").write_text('{"prompt":"Q","answer":"A"}\n')

    runs_dir = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic"
    _make_metrics(
        runs_dir / "50_1337",
        "episodic",
        {"em": 1.0},
        {"total_tokens": 1, "rss_mb": 1.0, "time_ms_per_100": 1.0},
    )
    base = tmp_path / "runs" / "20250101"
    summary = summarise(collect_metrics(base))
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))
    out_dir = tmp_path / "reports" / "20250101"
    write_smoke(data_root, out_dir / "smoke.md")
    write_reports(summary, retrieval, gates, out_dir, plots=False)
    idx_text = (out_dir / "index.md").read_text()
    assert "[smoke.md](smoke.md)" in idx_text
    smoke_text = (out_dir / "smoke.md").read_text()
    assert "## episodic" in smoke_text
    assert "| Q | A |" in smoke_text
