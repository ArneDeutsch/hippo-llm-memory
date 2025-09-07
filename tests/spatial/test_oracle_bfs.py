from types import SimpleNamespace

from hippo_eval.metrics.scoring import normalize_udlr, spatial_kpis


def test_udlr_validator():
    ok, bad = normalize_udlr("UDLR")
    assert ok == "UDLR" and not bad
    ok, bad = normalize_udlr("UDX")
    assert ok == "" and bad


def test_oracle_bfs():
    prompt = (
        "Grid 3x3 with obstacles []. Start (0, 0) goal (2, 0). "
        "Respond with moves using U, D, L, R only, no spaces (e.g., UDLR)."
    )
    task = SimpleNamespace(prompt=prompt)
    rows = [{"pred": "RR", "normalized_pred": "RR", "format_violation": 0}]
    metrics = spatial_kpis([task], rows)
    row = rows[0]
    assert row["oracle_path"] == "RR"
    assert row["oracle_success"]
    assert row["pred_matches_oracle"]
    assert metrics["oracle_success_rate"] == 1.0
