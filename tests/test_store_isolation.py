"""Ensure baseline runs do not create stores on disk."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from hippo_mem.eval.harness import _apply_model_defaults, evaluate
from hippo_mem.utils.stores import derive


def test_baseline_store_absent(tmp_path: Path, monkeypatch) -> None:
    """Running a baseline preset must not write any store files."""

    monkeypatch.setenv("RUN_ID", "foo")
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "eval" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)
    monkeypatch.chdir(tmp_path)
    data_file = Path("data/episodic/50_1337.jsonl")
    data_file.parent.mkdir(parents=True)
    data_file.write_text('{"prompt": "q", "answer": "a"}\n')
    cfg.suite = "episodic"
    cfg.preset = "baselines/core"
    cfg.model = str(Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2")
    cfg.n = 1
    cfg.seed = 1337
    cfg.dry_run = True
    cfg.persist = True
    layout = derive(run_id="foo", algo="hei_nw")
    cfg.store_dir = str(layout.algo_dir)
    cfg.session_id = layout.session_id
    cfg = _apply_model_defaults(cfg)

    outdir = tmp_path / "run"
    evaluate(cfg, outdir)

    assert not layout.algo_dir.exists()
    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics["store"]["size"] == 0
