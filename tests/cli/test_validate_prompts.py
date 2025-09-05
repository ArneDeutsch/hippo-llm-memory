import json
import sys
from pathlib import Path

import pytest

import scripts.validate_prompts as vp


def run_validator(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_prompts", *args])
    vp.main()


def test_validate_prompts_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    teach = tmp_path / "teach.jsonl"
    test = tmp_path / "test.jsonl"
    teach.write_text(json.dumps({"prompt": "fact one"}) + "\n")
    test.write_text(json.dumps({"prompt": "question?"}) + "\n")
    run_validator(monkeypatch, [str(teach), str(test)])


def test_validate_prompts_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    teach = tmp_path / "teach.jsonl"
    test = tmp_path / "test.jsonl"
    teach.write_text(json.dumps({"prompt": "fact"}) + "\n")
    test.write_text(json.dumps({"prompt": "fact?"}) + "\n")
    with pytest.raises(SystemExit):
        run_validator(monkeypatch, [str(teach), str(test)])
