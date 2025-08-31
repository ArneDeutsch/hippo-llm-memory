import pytest
from omegaconf import OmegaConf

from hippo_mem.utils.stores import assert_store_exists
from scripts.store_paths import derive


@pytest.mark.parametrize(
    "algo,kind", [("hei_nw", "episodic"), ("sgc_rss", "kg"), ("smpd", "spatial")]
)
def test_assert_store_exists_ok(tmp_path, algo, kind):
    base = tmp_path
    sid = "s"
    store = base / algo / sid / f"{kind}.jsonl"
    store.parent.mkdir(parents=True)
    store.write_text("{}")
    path = assert_store_exists(str(base), sid, algo, kind=kind)
    assert path == store


def test_assert_store_exists_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        assert_store_exists(str(tmp_path), "s", "hei_nw")


@pytest.mark.parametrize("algo,prefix", [("hei_nw", "hei"), ("sgc_rss", "sgc"), ("smpd", "smpd")])
def test_derive_store_layout(monkeypatch, tmp_path, algo, prefix):
    monkeypatch.chdir(tmp_path)
    layout = derive(run_id="foo", algo=algo)
    assert str(layout.base_dir) == "runs/foo/stores"
    assert str(layout.algo_dir) == f"runs/foo/stores/{algo}"
    assert layout.session_id == f"{prefix}_foo"


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize(
    "preset,algo,kind",
    [
        ("memory/hei_nw", "hei_nw", "episodic"),
        ("memory/sgc_rss", "sgc_rss", "kg"),
        ("memory/smpd", "smpd", "spatial"),
    ],
)
def test_eval_model_store_dir_normalization(tmp_path, monkeypatch, append, preset, algo, kind):
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
    assert called["assert_args"] == (str(base), "sid", algo, kind)
