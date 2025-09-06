import subprocess
import sys

from scripts import eval_cli


def test_explicit_overrides_ignore_env(monkeypatch):
    monkeypatch.setenv("HEI_SESSION_ID", "env_sid")
    monkeypatch.setenv("STORES", "/env/stores")
    monkeypatch.setenv("RUN_ID", "env_run")

    captured = {}

    def fake_call(cmd):  # pragma: no cover - patched in test
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_cli.py",
            "suite=semantic_mem",
            "session_id=cli_sid",
            "store_dir=/cli/stores",
            "run_id=cli_run",
        ],
    )

    assert eval_cli.main() == 0
    cmd = " ".join(captured["cmd"])
    assert cmd.count("session_id=cli_sid") == 1
    assert "env_sid" not in cmd
    assert cmd.count("store_dir=/cli/stores") == 1
    assert "/env/stores" not in cmd
    assert cmd.count("run_id=cli_run") == 1
    assert "env_run" not in cmd
