"""Ensure `Task` remains importable from the old location."""

from hippo_eval.eval.harness import Task as HarnessTask
from hippo_eval.eval.types import Task as TypesTask


def test_task_reexport() -> None:
    assert HarnessTask is TypesTask
