import sys

import pytest

import scripts.validate_store as vs
from hippo_mem.utils.stores import derive


def run_validator(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["validate_store", *args])
    vs.main()


def test_validate_store_ok(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_ID", "foo")
    layout = derive(run_id="foo", algo="hei_nw")
    store = layout.algo_dir / layout.session_id / "episodic.jsonl"
    store.parent.mkdir(parents=True)
    store.write_text("{}")
    (store.parent / "store_meta.json").write_text("{}")
    run_validator(monkeypatch, ["--algo", "hei_nw"])
    out = capsys.readouterr().out
    assert "OK:" in out


def test_validate_store_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_ID", "foo")
    with pytest.raises(SystemExit) as exc:
        run_validator(monkeypatch, ["--algo", "hei_nw"])
    assert exc.value.code == 1
