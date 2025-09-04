import sys

import pytest

import scripts.validate_store as vs


def run_validator(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["validate_store", *args])
    vs.main()


def test_validate_store_requires_run_id(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUN_ID", raising=False)
    with pytest.raises(SystemExit) as exc:
        run_validator(monkeypatch, ["--algo", "hei_nw"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "RUN_ID is required" in err
