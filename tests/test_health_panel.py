from __future__ import annotations

from pathlib import Path

from hippo_eval.reporting.health import Badge, render_panel
from hippo_eval.reporting.report import (
    collect_gate_ablation,
    collect_gates,
    collect_lineage,
    collect_metrics,
    collect_retrieval,
    summarise,
    summarise_gates,
    summarise_retrieval,
    write_reports,
)
from tests.test_report import _make_metrics


def test_panel_ordering() -> None:
    badges = [
        Badge("good", True),
        Badge("bad", False),
        Badge("warn", None),
    ]
    rendered = render_panel(badges)
    assert (
        rendered.index("bad-red")
        < rendered.index("warn-yellow")
        < rendered.index("good-brightgreen")
    )


def test_index_health_panel(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101"
    metrics_dir = base / "memory" / "hei_nw" / "episodic" / "50_1337"
    gates = {"episodic": {"attempts": 0}}
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
    _make_metrics(
        metrics_dir,
        "episodic",
        {"post_em": 0.5},
        gates=gates,
        retrieval=retrieval,
        store={"size": 1, "source": "replay"},
        bench=True,
    )
    summary, missing_pre = summarise(collect_metrics(base))
    retrieval_s = summarise_retrieval(collect_retrieval(base))
    gates_s = summarise_gates(collect_gates(base))
    gate_ablation = collect_gate_ablation(base)
    lineage = collect_lineage(base)
    out = tmp_path / "reports" / "20250101"
    write_reports(
        summary,
        retrieval_s,
        gates_s,
        gate_ablation,
        out,
        plots=False,
        seed_count=1,
        lineage=lineage,
        missing_pre=missing_pre,
    )
    line = (out / "index.md").read_text().splitlines()[2]
    parts = line.split()
    assert parts[0].startswith("[![BaselinesOK-red]")
    assert parts[1].startswith("[![GatingActive-red]")
    assert parts[2].startswith("[![NonStubStores-brightgreen]")
    assert any("RetrievalActive-brightgreen" in p for p in parts)
