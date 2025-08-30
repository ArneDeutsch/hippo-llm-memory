import pytest
from omegaconf import OmegaConf

from hippo_mem.utils.stores import assert_store_exists
from scripts.store_paths import derive


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


def test_derive_store_layout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    layout = derive(run_id="foo", algo="sgc_rss")
    assert str(layout.base_dir) == "runs/foo/stores"
    assert str(layout.algo_dir) == "runs/foo/stores/sgc_rss"
    assert layout.session_id == "sgc_rss_foo"


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize(
    "preset,algo", [("memory/hei_nw", "hei_nw"), ("memory/sgc_rss", "sgc_rss")]
)
def test_eval_model_store_dir_normalization(tmp_path, monkeypatch, append, preset, algo):
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

    monkeypatch.setattr(em, "assert_store_exists", fake_assert)
    monkeypatch.setattr(em, "harness_main", fake_harness)

    em.main.__wrapped__(cfg)

    expected = store_dir if append else base / algo
    assert called["cfg_store_dir"] == str(expected)
    assert called["assert_args"] == (str(base), "sid", algo, "episodic")
