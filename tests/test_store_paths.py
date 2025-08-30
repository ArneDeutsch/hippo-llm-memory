from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.eval_model as eval_model
from hippo_mem.utils.stores import assert_store_exists


def test_assert_store_exists(tmp_path: Path) -> None:
    base = tmp_path
    sid = "s1"
    store_file = base / "hei_nw" / sid / "episodic.jsonl"
    store_file.parent.mkdir(parents=True)
    store_file.write_text("{}\n")

    path = assert_store_exists(str(base), sid)
    assert path == store_file

    with pytest.raises(FileNotFoundError):
        assert_store_exists(str(base), "missing")


def test_eval_model_store_dir_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "stores"
    sid = "abc"
    expected = str(base / "hei_nw")

    captured: dict[str, str] = {}

    def fake_assert(store_dir: str, session_id: str, kind: str = "episodic") -> Path:
        captured["assert_store_dir"] = store_dir
        captured["assert_sid"] = session_id
        return Path(store_dir) / "hei_nw" / session_id / f"{kind}.jsonl"

    def fake_harness(cfg) -> None:
        captured["cfg_store_dir"] = cfg.store_dir
        captured["cfg_sid"] = cfg.session_id

    monkeypatch.setattr("hippo_mem.utils.stores.assert_store_exists", fake_assert)
    monkeypatch.setattr("scripts.eval_model.harness_main", fake_harness)

    for input_dir in [base, base / "hei_nw"]:
        cfg = OmegaConf.create({"mode": "test", "store_dir": str(input_dir), "session_id": sid})
        captured.clear()
        eval_model.main.__wrapped__(cfg)
        assert captured["assert_store_dir"] == str(base)
        assert captured["cfg_store_dir"] == expected
