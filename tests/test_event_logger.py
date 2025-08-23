from __future__ import annotations

import json

from hippo_mem.episodic.event_logger import EventLogger


def test_log_event_writes_jsonl(tmp_path) -> None:
    """log_event writes JSON lines to the configured file."""

    log_file = tmp_path / "events.jsonl"
    logger = EventLogger(str(log_file))
    logger.log_event("write", {"id": 1})
    logger.log_event("hit", {"id": 2})

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    assert events[0]["op"] == "write"
    assert events[1]["op"] == "hit"
