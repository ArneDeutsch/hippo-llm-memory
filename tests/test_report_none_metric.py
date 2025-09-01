import json
from pathlib import Path

from scripts.report import (
    collect_gate_ablation,
    collect_gates,
    collect_metrics,
    collect_retrieval,
    summarise,
    summarise_gates,
    summarise_retrieval,
    write_reports,
)


def test_report_handles_null_em_raw(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "20250101" / "baselines" / "core" / "semantic" / "50_1337"
    base.mkdir(parents=True)
    content = {"metrics": {"semantic": {"em_raw": None}}}
    (base / "metrics.json").write_text(json.dumps(content))

    runs_root = tmp_path / "runs" / "20250101"
    summary, _ = summarise(collect_metrics(runs_root))
    retrieval = summarise_retrieval(collect_retrieval(runs_root))
    gates = summarise_gates(collect_gates(runs_root))
    gate_ablation = collect_gate_ablation(runs_root)
    out_dir = tmp_path / "reports" / "20250101"
    write_reports(summary, retrieval, gates, gate_ablation, out_dir, plots=False, seed_count=1)
    assert (out_dir / "semantic" / "summary.md").exists()
