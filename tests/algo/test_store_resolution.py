from hippo_eval.eval.store_utils import resolve_store_meta_path


def test_store_meta_path_resolution_base_and_algo_dir(tmp_path):
    base = tmp_path / "runs" / "rid" / "stores"
    algo = "hei_nw"
    session = "hei_rid"
    # Base directory style
    path_base = resolve_store_meta_path("memory/hei_nw", base, session)
    assert path_base == base / algo / session / "store_meta.json"
    # Algorithm directory style
    algo_dir = base / algo
    path_algo = resolve_store_meta_path("memory/hei_nw", algo_dir, session)
    assert path_algo == algo_dir / session / "store_meta.json"
