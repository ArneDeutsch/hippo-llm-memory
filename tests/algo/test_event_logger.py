# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import json

from hippo_mem.common import ProvenanceLogger
from hippo_mem.episodic.event_logger import EventLogger


def test_log_event_writes_jsonl(tmp_path) -> None:
    """log_event writes JSON lines to the configured file."""

    log_file = tmp_path / "events.jsonl"
    logger = EventLogger(str(log_file))
    logger.log_event("write", {"payload": {"id": 1}})
    logger.log_event("hit", {"payload": {"id": 2}})

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    for event in events:
        assert {"ts", "op", "payload"}.issubset(event)
    assert events[0]["op"] == "write"
    assert events[1]["op"] == "hit"


def test_event_logger_provenance_roundtrip(tmp_path) -> None:
    """EventLogger payloads can feed ProvenanceLogger without losing structure."""

    event_log = tmp_path / "events.jsonl"
    logger = EventLogger(str(event_log))
    payload = {"id": 7}
    logger.log_event("write", {"payload": payload})

    event = json.loads(event_log.read_text(encoding="utf-8").splitlines()[0])

    prov_logger = ProvenanceLogger(str(tmp_path))
    prov_logger.log(
        mem="episodic",
        action=event["op"],
        reason="roundtrip",
        payload=event["payload"],
    )

    rec = json.loads((tmp_path / "provenance.ndjson").read_text(encoding="utf-8").splitlines()[0])
    assert {"ts", "action", "payload"}.issubset(rec)
    assert rec["action"] == "write"
    assert rec["payload"] == payload
