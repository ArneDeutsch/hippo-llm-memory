import pytest  # noqa: F401
from omegaconf import OmegaConf  # noqa: F401

from hippo_mem.utils.stores import assert_store_exists


@pytest.mark.parametrize("algo", ["hei_nw", "sgc_rss"])
def test_assert_store_exists_ok(tmp_path, algo):
    base = tmp_path
    sid = "s"
    store = base / algo / sid / "episodic.jsonl"
    store.parent.mkdir(parents=True)
    store.write_text("{}")
    path = assert_store_exists(str(base), sid, algo)
    assert path == store


def test_assert_store_exists_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        assert_store_exists(str(tmp_path), "s", "hei_nw")


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize(
    "preset,algo", [("memory/hei_nw", "hei_nw"), ("memory/sgc_rss", "sgc_rss")]
)
def test_eval_model_store_dir_normalization(tmp_path, monkeypatch, capsys, append, preset, algo):
    import hippo_mem.utils.stores as stores
    import scripts.eval_model as em

    base = tmp_path / "base"
    base.mkdir()
    store_dir = base / algo if append else base
    cfg = OmegaConf.create(
        {"mode": "test", "store_dir": str(store_dir), "session_id": "sid", "preset": preset}
    )

    called = {}

    def fake_assert(sd, sid, al, kind="episodic"):
        called["assert_args"] = (sd, sid, al, kind)

    def fake_harness(cfg):
        called["cfg_store_dir"] = cfg.store_dir

    monkeypatch.setattr(stores, "assert_store_exists", fake_assert)
    monkeypatch.setattr(em, "harness_main", fake_harness)

    em.main.__wrapped__(cfg)

    expected = store_dir if append else base / algo
    assert called["cfg_store_dir"] == str(expected)
    assert called["assert_args"] == (str(base), "sid", algo, "episodic")

    err = capsys.readouterr().err
    if append:
        assert f"already ends with '{algo}'" in err
    else:
        assert "Warning" not in err
