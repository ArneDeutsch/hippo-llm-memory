import pytest  # noqa: F401
from omegaconf import OmegaConf  # noqa: F401

from hippo_mem.utils.stores import assert_store_exists


def test_assert_store_exists_ok(tmp_path):
    base = tmp_path
    sid = "s"
    store = base / "hei_nw" / sid / "episodic.jsonl"
    store.parent.mkdir(parents=True)
    store.write_text("{}")
    path = assert_store_exists(str(base), sid)
    assert path == store


def test_assert_store_exists_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        assert_store_exists(str(tmp_path), "s")


@pytest.mark.parametrize("append", [False, True])
def test_eval_model_store_dir_normalization(tmp_path, monkeypatch, capsys, append):
    import hippo_mem.utils.stores as stores
    import scripts.eval_model as em

    base = tmp_path / "base"
    base.mkdir()
    store_dir = base / "hei_nw" if append else base
    cfg = OmegaConf.create({"mode": "test", "store_dir": str(store_dir), "session_id": "sid"})

    called = {}

    def fake_assert(sd, sid, kind="episodic"):
        called["assert_args"] = (sd, sid, kind)

    def fake_harness(cfg):
        called["cfg_store_dir"] = cfg.store_dir

    monkeypatch.setattr(stores, "assert_store_exists", fake_assert)
    monkeypatch.setattr(em, "harness_main", fake_harness)

    em.main.__wrapped__(cfg)

    expected = store_dir if append else base / "hei_nw"
    assert called["cfg_store_dir"] == str(expected)
    assert called["assert_args"] == (str(base), "sid", "episodic")

    err = capsys.readouterr().err
    if append:
        assert "Warning: store_dir already ends with 'hei_nw'; not appending." in err
    else:
        assert "Warning" not in err
