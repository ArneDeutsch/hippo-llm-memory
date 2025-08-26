from hippo_mem.common.history import HistoryEntry, RollbackMixin


class DummyHistory(RollbackMixin):
    """Simple class to track integer appends and support rollbacks."""

    def __init__(self, max_undo: int = 3) -> None:
        super().__init__(max_undo)
        self.state: list[int] = []

    def _apply_rollback(self, entry: HistoryEntry) -> None:
        self.state.pop()


def test_rollback_restores_state_and_truncates_history() -> None:
    dummy = DummyHistory(max_undo=3)

    for i in range(5):
        dummy.state.append(i)
        dummy._push_history("add", i)

    assert dummy.state == [0, 1, 2, 3, 4]
    assert [h.data for h in dummy._history] == [2, 3, 4]

    dummy.rollback()
    assert dummy.state == [0, 1, 2, 3]
    assert [h.data for h in dummy._history] == [2, 3]

    dummy.rollback(2)
    assert dummy.state == [0, 1]
    assert dummy._history == []

    dummy.rollback(5)
    assert dummy.state == [0, 1]
