import json
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


def test_validate_store_strict_injected_context(tmp_path, monkeypatch):
    """Strict mode fails when injected context is missing."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_ID", "foo")
    layout = derive(run_id="foo", algo="hei_nw")
    store = layout.algo_dir / layout.session_id / "episodic.jsonl"
    store.parent.mkdir(parents=True)
    store.write_text(
        json.dumps(
            {"schema": "episodic.v1", "id": 1, "key": [1.0], "value": {}, "ts": 0, "salience": 0}
        )
        + "\n"
    )
    (store.parent / "store_meta.json").write_text("{}")
    run_dir = tmp_path / "hei_nw" / "suite" / "50_1"
    run_dir.mkdir(parents=True)
    metrics = {
        "n": 1,
        "retrieval": {"episodic": {"requests": 1}},
        "store": {"per_memory": {"episodic": 1}},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "audit_sample.jsonl").write_text(
        json.dumps({"id": 0, "router_path": ["episodic"]}) + "\n"
    )
    with pytest.raises(SystemExit):
        run_validator(
            monkeypatch,
            ["--algo", "hei_nw", "--metrics", str(run_dir / "metrics.json"), "--strict-telemetry"],
        )
    (run_dir / "audit_sample.jsonl").write_text(
        json.dumps({"id": 0, "router_path": ["episodic"], "injected_context": ["ctx"]}) + "\n"
    )
    run_validator(
        monkeypatch,
        ["--algo", "hei_nw", "--metrics", str(run_dir / "metrics.json"), "--strict-telemetry"],
    )
