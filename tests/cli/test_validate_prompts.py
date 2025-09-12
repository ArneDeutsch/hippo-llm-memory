# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import json
import sys
from pathlib import Path

import pytest

import scripts.validate_prompts as vp


def run_validator(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_prompts", *args])
    vp.main()


def test_validate_prompts_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    teach = tmp_path / "teach.jsonl"
    test = tmp_path / "test.jsonl"
    teach.write_text(json.dumps({"prompt": "fact one"}) + "\n")
    test.write_text(json.dumps({"prompt": "question?"}) + "\n")
    run_validator(monkeypatch, [str(teach), str(test)])
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
    assert vp._load_tokens(teach) == {"fact", "one"}


def test_validate_prompts_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    teach = tmp_path / "teach.jsonl"
    test = tmp_path / "test.jsonl"
    teach.write_text(json.dumps({"prompt": "fact"}) + "\n")
    test.write_text(json.dumps({"prompt": "fact?"}) + "\n")
    with pytest.raises(SystemExit):
        run_validator(monkeypatch, [str(teach), str(test)])
