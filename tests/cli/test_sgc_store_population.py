import subprocess
import sys


def test_sgc_teach_populates_store(tmp_path):
    outdir = tmp_path / "o"
    store = tmp_path / "s"
    sid = "sgc_test"
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "suite=semantic_mem",
        "preset=memory/sgc_rss",
        "n=1",
        "seed=1337",
        f"outdir={outdir}",
        f"store_dir={store}",
        f"session_id={sid}",
        "model=models/tiny-gpt2",
        "mode=teach",
        "persist=true",
    ]
    subprocess.check_call(cmd)
    kg = store / "sgc_rss" / sid / "kg.jsonl"
    meta = store / "sgc_rss" / sid / "store_meta.json"
    assert meta.exists()
    assert kg.exists()
    # must have at least one non-empty JSON record
    assert any(line.strip() for line in kg.read_text().splitlines())
