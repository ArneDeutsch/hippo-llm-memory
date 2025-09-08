from hippo_eval.eval.types import Task
from hippo_eval.metrics.scoring import enforce_udlr, spatial_multi_kpis


def test_enforce_udlr_and_oracle_path(monkeypatch) -> None:
    monkeypatch.setenv("HIPPO_ENFORCE_UDLR", "1")
    monkeypatch.setenv("HIPPO_ORACLE", "1")
    assert enforce_udlr("UDLR") == "UDLR"
    assert enforce_udlr("XYZ") == ""
    task = Task(
        prompt="Grid 3x3 with obstacles []. Start (0, 0) goal (0, 2). Respond with moves using U, D, L, R only, no spaces (e.g., UDLR).",
        answer="DD",
    )
    rows = [
        {
            "prompt": task.prompt,
            "answer": task.answer,
            "pred": "DD",
            "normalized_pred": "DD",
        }
    ]
    metrics = spatial_multi_kpis([task], rows)
    assert rows[0]["oracle_path"] == "DD"
    assert rows[0]["oracle_success"] is True
    assert rows[0]["pred_matches_oracle"] is True
    assert metrics["oracle_success_rate"] == 1.0
    assert metrics["valid_action_rate"] == 1.0
