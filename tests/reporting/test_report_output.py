# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import json
from pathlib import Path

from hippo_eval.reporting.report import write_reports
from hippo_eval.reporting.rollup import (
    collect_gate_ablation,
    collect_gates,
    collect_metrics,
    collect_retrieval,
    summarise,
    summarise_gates,
    summarise_retrieval,
)


def test_report_shows_em_and_diagnostics(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101" / "baselines" / "core" / "episodic" / "50_1337"
    base.mkdir(parents=True)
    content = {
        "n": 10,
        "metrics": {
            "episodic": {
                "em_raw": 0.5,
                "em_norm": 0.6,
                "f1": 0.7,
            },
            "compute": {},
        },
        "diagnostics": {"episodic": {"overlong": 2, "format_violation": 1}},
    }
    (base / "metrics.json").write_text(json.dumps(content))

    runs = tmp_path / "runs" / "20250101"
    metrics = collect_metrics(runs)
    summary, _ = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(runs))
    gates = summarise_gates(collect_gates(runs))
    gate_ablation = collect_gate_ablation(runs)
    out = tmp_path / "reports" / "20250101"
    paths = write_reports(summary, retrieval, gates, gate_ablation, out, plots=False, seed_count=1)
    text = paths["episodic"].read_text()
    assert "EM (raw)" in text and "EM (norm)" in text
    assert "overlong" in text and "format_violation" in text
    # ensure ratios computed: 2/10=0.2, 1/10=0.1
    assert "0.200" in text and "0.100" in text


def test_report_retrieval_section(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101" / "memory" / "hei_nw" / "episodic" / "50_1337"
    base.mkdir(parents=True)
    content = {
        "metrics": {"episodic": {"em": 1.0}},
        "retrieval": {
            "episodic": {
                "k": 2,
                "batch_size": 1,
                "requests": 2,
                "hits_at_k": 1,
                "hit_rate_at_k": 0.25,
                "tokens_returned": 4,
                "avg_latency_ms": 0.1,
            }
        },
    }
    (base / "metrics.json").write_text(json.dumps(content))

    runs = tmp_path / "runs" / "20250101"
    metrics = collect_metrics(runs)
    summary, _ = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(runs))
    gates = summarise_gates(collect_gates(runs))
    gate_ablation = collect_gate_ablation(runs)
    out = tmp_path / "reports" / "20250101"
    write_reports(summary, retrieval, gates, gate_ablation, out, plots=False, seed_count=1)
    text = (out / "episodic" / "summary.md").read_text()
    header = "| mem | k | batch_size | requests | hits_at_k | hit_rate_at_k | tokens_returned | avg_latency_ms |"
    assert header in text
    assert "| episodic | 2 | 1 | 2 | 1 | 0.250 | 4 | 0.100 |" in text
    assert "actual recalled traces" in text


def test_no_retrieval_badge(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101" / "memory" / "hei_nw" / "episodic" / "50_1337"
    base.mkdir(parents=True)
    content = {
        "metrics": {"episodic": {"pre_em_norm": 0.1}},
        "retrieval": {
            "episodic": {
                "k": 1,
                "batch_size": 1,
                "requests": 0,
                "hits_at_k": 0,
                "hit_rate_at_k": 0.0,
                "tokens_returned": 0,
                "avg_latency_ms": 0.0,
            }
        },
        "store": {"size": 0},
    }
    (base / "metrics.json").write_text(json.dumps(content))

    runs = tmp_path / "runs" / "20250101"
    metrics = collect_metrics(runs)
    summary, _ = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(runs))
    gates = summarise_gates(collect_gates(runs))
    gate_ablation = collect_gate_ablation(runs)
    out = tmp_path / "reports" / "20250101"
    write_reports(summary, retrieval, gates, gate_ablation, out, plots=False, seed_count=1)
    text = (out / "episodic" / "summary.md").read_text()
    assert "NoRetrieval" in text


def test_missing_pre_marked(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101" / "memory" / "hei_nw" / "episodic" / "50_1337"
    base.mkdir(parents=True)
    content = {"metrics": {"episodic": {"post_em": 0.5}}}
    (base / "metrics.json").write_text(json.dumps(content))

    runs = tmp_path / "runs" / "20250101"
    metrics = collect_metrics(runs)
    summary, _ = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(runs))
    gates = summarise_gates(collect_gates(runs))
    gate_ablation = collect_gate_ablation(runs)
    out = tmp_path / "reports" / "20250101"
    write_reports(summary, retrieval, gates, gate_ablation, out, plots=False, seed_count=1)
    text = (out / "episodic" / "summary.md").read_text()
    assert "_missing_" in text
    assert "MissingPre" in text
