# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from hippo_eval.eval.modes import (
    Mode,
    ReplayStrategy,
    TeachStrategy,
    TestStrategy,
    get_mode_strategy,
)


def test_mode_enum_values() -> None:
    assert Mode.TEACH.value == "teach"
    assert Mode.TEST.value == "test"
    assert Mode.REPLAY.value == "replay"


def test_strategy_flags() -> None:
    teach = get_mode_strategy(Mode.TEACH)
    assert isinstance(teach, TeachStrategy)
    assert not teach.retrieval_enabled
    assert teach.ingest_enabled

    test = get_mode_strategy(Mode.TEST)
    assert isinstance(test, TestStrategy)
    assert test.retrieval_enabled
    assert not test.ingest_enabled

    replay = get_mode_strategy(Mode.REPLAY)
    assert isinstance(replay, ReplayStrategy)
    assert replay.retrieval_enabled
    assert not replay.ingest_enabled
