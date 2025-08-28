from hippo_mem.eval.harness import Task
from hippo_mem.eval.score import spatial_kpis


def test_spatial_kpis_basic() -> None:
    tasks = [
        Task(
            prompt="Grid 3x3 with obstacles []. Start (0, 0) goal (0, 2). What is the shortest path length?",
            answer="2",
        ),
        Task(
            prompt="Grid 3x3 with obstacles []. What move sequence leads from (0, 0) to (0, 2)?",
            answer="DD",
        ),
    ]
    rows = [
        {"prompt": tasks[0].prompt, "answer": tasks[0].answer, "pred": "2"},
        {"prompt": tasks[1].prompt, "answer": tasks[1].answer, "pred": "DD"},
    ]
    metrics = spatial_kpis(tasks, rows)
    assert metrics["success_rate"] == 1.0
    assert metrics["suboptimality_ratio"] == 1.0
    assert metrics["steps_to_goal"] == 2.0
    assert rows[1]["steps_pred"] == 2
    assert rows[1]["steps_opt"] == 2
