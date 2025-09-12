# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from pathlib import Path

from hippo_mem.consolidation.trainer import Args, load_config, train


def test_train_handles_float_token_ids(tmp_path: Path) -> None:
    store_root = tmp_path / "stores" / "hei_nw"
    session_id = "s1"
    session_dir = store_root / session_id
    session_dir.mkdir(parents=True)
    # minimal episodic store with one record
    (session_dir / "episodic.jsonl").write_text(
        '{"prompt": "hi", "answer": "there"}\n', encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    args = Args(
        store_dir=str(store_root),
        session_id=session_id,
        outdir=str(out_dir),
        model="models/tiny-gpt2",
        config=None,
    )
    cfg = load_config(None)
    cfg["train"]["steps"] = 1
    cfg["train"]["batch_size"] = 1
    cfg["replay"]["cycles"] = 1

    train(args, cfg)

    assert (out_dir / "adapter_config.json").exists()
