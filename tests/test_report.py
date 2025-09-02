from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.report import (
    _find_latest_run_id,
    _missing_pre_suites,
    collect_gate_ablation,
    collect_gates,
    collect_lineage,
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
    retrieval: dict | None = None,
    store: dict | None = None,
    dataset_profile: str | None = None,
    bench: bool = False,
    seed: int = 1337,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    content = {"metrics": {suite: metrics}}
    if bench:
        content.update({"suite": suite, "preset": "baselines/core", "n": 50, "seed": seed})
    if compute:
        content["metrics"]["compute"] = compute
    if gates:
        content["gating"] = gates
    if retrieval:
        content["retrieval"] = retrieval
    if store:
        content["store"] = store
    if dataset_profile is not None:
        content["dataset_profile"] = dataset_profile
    (path / "metrics.json").write_text(json.dumps(content))


def test_report_aggregation(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs" / "20250101" / "baselines" / "core"
    runs_on = base_dir / "gate_on" / "episodic"
    runs_off = base_dir / "gate_off" / "episodic"
    gates_on = {
        "relational": {
            "attempts": 10,
            "accepted": 10,
            "inserted": 8,
            "aggregated": 2,
            "routed_to_episodic": 0,
        },
        "spatial": {
            "attempts": 20,
            "accepted": 20,
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
            "accepted": 10,
            "inserted": 10,
            "aggregated": 0,
            "routed_to_episodic": 0,
        },
        "spatial": {
            "attempts": 20,
            "accepted": 20,
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
        {
            "input_tokens": 5,
            "generated_tokens": 5,
            "total_tokens": 10,
            "rss_mb": 1.0,
            "time_ms_per_100": 2.0,
            "latency_ms_mean": 1.0,
        },
        gates_on,
        bench=True,
        seed=1337,
    )
    _make_metrics(
        runs_on / "50_2025",
        "episodic",
        {"em_raw": 0.7, "r": 0.9},
        {
            "input_tokens": 15,
            "generated_tokens": 15,
            "total_tokens": 30,
            "rss_mb": 1.5,
            "time_ms_per_100": 4.0,
            "latency_ms_mean": 1.0,
        },
        gates_on,
        bench=True,
        seed=2025,
    )
    _make_metrics(
        runs_off / "50_4242",
        "episodic",
        {"em_raw": 0.6, "r": 0.8},
        {
            "input_tokens": 10,
            "generated_tokens": 10,
            "total_tokens": 20,
            "rss_mb": 2.0,
            "time_ms_per_100": 3.0,
            "latency_ms_mean": 1.0,
        },
        gates_off,
        bench=True,
        seed=4242,
    )
    # also create another suite to ensure per-suite report generation
    other = tmp_path / "runs" / "20250101" / "baselines" / "core" / "semantic"
    _make_metrics(
        other / "50_1337",
        "semantic",
        {"post_em": 0.4},
        {
            "input_tokens": 2,
            "generated_tokens": 3,
            "total_tokens": 5,
            "rss_mb": 0.5,
            "time_ms_per_100": 1.0,
            "latency_ms_mean": 1.0,
        },
        bench=True,
        seed=1337,
    )

    base = tmp_path / "runs" / "20250101"
    metrics = collect_metrics(base)
    for recs in metrics.values():
        for rec in recs:
            for key in (
                "input_tokens",
                "generated_tokens",
                "total_tokens",
                "time_ms_per_100",
                "rss_mb",
                "latency_ms_mean",
            ):
                assert key in rec
    summary, _ = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))
    gate_ablation = collect_gate_ablation(base)
    lineage = collect_lineage(base)

    em_stats = summary["episodic"]["baselines/core/gate_on"][50]["em_raw"]
    assert em_stats[0] == 0.6
    assert round(em_stats[1], 3) == 0.196

    out = tmp_path / "reports" / "20250101"
    paths = write_reports(
        summary,
        retrieval,
        gates,
        gate_ablation,
        out,
        plots=False,
        seed_count=1,
        lineage=lineage,
    )
    # per-suite summaries present
    assert set(paths.keys()) == {"episodic", "semantic"}
    md_path = paths["episodic"]
    text = md_path.read_text()
    # table header contains all fields
    header = next(line for line in text.splitlines() if line.startswith("| Preset"))
    for col in [
        "EM (raw)",
        "r",
        "rss_mb",
        "time_ms_per_100",
        "total_tokens",
    ]:
        assert col in header
    # both presets appear as rows
    assert "| baselines/core/gate_on | 50 |" in text
    assert "| baselines/core/gate_off | 50 |" in text
    assert "seeds:1337,2025,4242" in text
    # gate telemetry rendered
    assert "duplicate_rate" in text
    assert "nodes_per_1k" in text
    assert "Gate ON vs OFF" in text
    assert "| mem | store_on | store_off | accepted_on | accepted_off | ΔEM |" in text

    idx = out / "index.md"
    assert idx.exists()
    idx_text = idx.read_text()
    header = next(line for line in idx_text.splitlines() if line.startswith("| Suite"))
    for col in ["EM (raw)", "r", "rss_mb", "time_ms_per_100", "total_tokens"]:
        assert col in header
    assert "[episodic](episodic/summary.md)" in idx_text
    assert "[semantic](semantic/summary.md)" in idx_text
    assert "_missing_" in idx_text
    assert "MissingPre" in idx_text
    # gate telemetry roll-up present
    assert "## Gate Telemetry" in idx_text


def test_report_handles_missing_optional(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic"
    _make_metrics(base_dir / "50_1337", "episodic", {"em": 0.5}, {"total_tokens": 10})
    base = tmp_path / "runs" / "20250101"
    summary, _ = summarise(collect_metrics(base))
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))
    gate_ablation = collect_gate_ablation(base)
    out = tmp_path / "reports" / "20250101"
    paths = write_reports(summary, retrieval, gates, gate_ablation, out, plots=False, seed_count=1)
    text = paths["episodic"].read_text()
    assert "Retrieval Telemetry" not in text


def test_report_warnings(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101"
    _make_metrics(
        base / "baselines" / "core" / "episodic" / "50_1337",
        "episodic",
        {"pre_em_norm": 0.0},
        retrieval={
            "episodic": {
                "k": 1,
                "batch_size": 1,
                "requests": 1,
                "hits_at_k": 0,
                "hit_rate_at_k": 0.0,
                "tokens_returned": 0,
                "avg_latency_ms": 0.0,
            }
        },
        store={"size": 1},
    )
    _make_metrics(
        base / "memory" / "hei_nw" / "episodic" / "50_1337",
        "episodic",
        {"pre_em_norm": 1.0},
        gates={"episodic": {"attempts": 0}},
        retrieval={
            "episodic": {
                "k": 1,
                "batch_size": 1,
                "requests": 5,
                "hits_at_k": 0,
                "hit_rate_at_k": 0.0,
                "tokens_returned": 0,
                "avg_latency_ms": 0.0,
            }
        },
        store={"size": 5},
    )
    _make_metrics(
        base / "ablate" / "no_retrieval" / "episodic" / "50_1337",
        "episodic",
        {"pre_em_norm": 0.1},
        retrieval={
            "episodic": {
                "k": 1,
                "batch_size": 1,
                "requests": 2,
                "hits_at_k": 0,
                "hit_rate_at_k": 0.0,
                "tokens_returned": 0,
                "avg_latency_ms": 0.0,
            }
        },
        store={"size": 0},
    )
    summary, _ = summarise(collect_metrics(base))
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))
    gate_ablation = collect_gate_ablation(base)
    out = tmp_path / "reports" / "20250101"
    write_reports(summary, retrieval, gates, gate_ablation, out, plots=False, seed_count=1)
    text = (out / "episodic" / "summary.md").read_text()
    assert "BaselineTelemetry" in text
    assert "NoRetrievalTelemetry" in text
    assert "SaturationSuspect" in text
    assert "GateNoOp" in text
    assert "### Warnings" in text
    assert (out / "index.md").exists()


def test_find_latest_run_id(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    (base / "20240101").mkdir(parents=True)
    (base / "20250102").mkdir()
    assert _find_latest_run_id(base) == "20250102"


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
        {
            "input_tokens": 1,
            "generated_tokens": 0,
            "total_tokens": 1,
            "rss_mb": 1.0,
            "time_ms_per_100": 1.0,
            "latency_ms_mean": 1.0,
        },
    )
    base = tmp_path / "runs" / "20250101"
    summary, _ = summarise(collect_metrics(base))
    retrieval = summarise_retrieval(collect_retrieval(base))
    gates = summarise_gates(collect_gates(base))
    gate_ablation = collect_gate_ablation(base)
    out_dir = tmp_path / "reports" / "20250101"
    write_smoke(data_root, out_dir / "smoke.md")
    write_reports(summary, retrieval, gates, gate_ablation, out_dir, plots=False, seed_count=1)
    idx_text = (out_dir / "index.md").read_text()
    assert "[smoke.md](smoke.md)" in idx_text
    smoke_text = (out_dir / "smoke.md").read_text()
    assert "## episodic" in smoke_text
    assert "| Q | A |" in smoke_text


@pytest.mark.parametrize("telemetry", [False, True])
def test_report_handles_optional_telemetry(tmp_path: Path, telemetry: bool) -> None:
    """Report generation works for bench-style and harness metrics."""

    base = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic"
    retrieval = None
    gates = None
    if telemetry:
        retrieval = {
            "episodic": {
                "k": 1,
                "batch_size": 1,
                "requests": 1,
                "hits_at_k": 0,
                "hit_rate_at_k": 0.0,
                "tokens_returned": 0,
                "avg_latency_ms": 0.0,
            }
        }
        gates = {"episodic": {"attempts": 1}}
    _make_metrics(
        base / "50_1337",
        "episodic",
        {"em": 0.5},
        retrieval=retrieval,
        gates=gates,
        store={"size": 0},
        bench=not telemetry,
    )
    root = tmp_path / "runs" / "20250101"
    summary, _ = summarise(collect_metrics(root))
    retrieval_s = summarise_retrieval(collect_retrieval(root))
    gates_s = summarise_gates(collect_gates(root))
    gate_ablation = collect_gate_ablation(root)
    out_dir = tmp_path / "reports" / "20250101"
    write_reports(summary, retrieval_s, gates_s, gate_ablation, out_dir, plots=False, seed_count=1)
    text = (out_dir / "episodic" / "summary.md").read_text()
    if telemetry:
        assert "Retrieval Telemetry" in text or "Gate Telemetry" in text
    else:
        assert "Retrieval Telemetry" not in text
        assert "Gate Telemetry" not in text


def test_collect_lineage_tracks_dataset_profile_and_store_source(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101"
    metrics_dir = base / "baselines" / "core" / "episodic" / "50_1337"
    _make_metrics(
        metrics_dir,
        "episodic",
        {"em": 0.5},
        bench=True,
        store={"size": 0, "source": "replay"},
        dataset_profile="hard",
    )
    lineage = collect_lineage(base)
    assert lineage["episodic"]["profiles"] == {"hard"}
    assert lineage["episodic"]["store_source"] == {"replay"}


def test_missing_post_metrics_detector() -> None:
    data = {("episodic", "preset", 50): [{"pre_em": 0.1}]}
    from scripts.report import _missing_post_metrics

    missing = _missing_post_metrics(data)
    assert missing == [("episodic", "preset", 50)]


def test_missing_pre_suites_detector() -> None:
    data = {("episodic", "preset", 50): [{"post_em": 0.8}, {"post_em": 0.7, "pre_em": 0.6}]}
    _, missing_pre = summarise(data)
    assert _missing_pre_suites(missing_pre) == ["episodic"]
