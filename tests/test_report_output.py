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
    summary = summarise(metrics)
    retrieval = summarise_retrieval(collect_retrieval(runs))
    gates = summarise_gates(collect_gates(runs))
    out = tmp_path / "reports" / "20250101"
    paths = write_reports(summary, retrieval, gates, out, plots=False)
    text = paths["episodic"].read_text()
    assert "EM (raw)" in text and "EM (norm)" in text
    assert "overlong" in text and "format_violation" in text
    # ensure ratios computed: 2/10=0.2, 1/10=0.1
    assert "0.200" in text and "0.100" in text
