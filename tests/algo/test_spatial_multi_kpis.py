# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from hippo_eval.eval.types import Task
from hippo_eval.metrics.scoring import spatial_multi_kpis


def test_spatial_multi_kpis_learning_curve() -> None:
    tasks = [
        Task(
            prompt="Grid 3x3 with obstacles []. Start (0, 0) goal (0, 2). Respond with moves using U, D, L, R only, no spaces (e.g., UDLR).",
            answer="DD",
            episode_id="ep1",
        ),
        Task(
            prompt="Grid 3x3 with obstacles []. Start (0, 0) goal (0, 2). Respond with moves using U, D, L, R only, no spaces (e.g., UDLR).",
            answer="DD",
            episode_id="ep2",
        ),
    ]
    rows = [
        {"prompt": tasks[0].prompt, "answer": tasks[0].answer, "pred": "DD"},
        {"prompt": tasks[1].prompt, "answer": tasks[1].answer, "pred": "UU"},
    ]
    metrics = spatial_multi_kpis(tasks, rows)
    assert metrics["success_rate"] == 0.5
    assert metrics["mean_plan_length"] == 2.0
    assert metrics["success_ep1"] == 1.0
    assert metrics["success_ep2"] == 0.0
