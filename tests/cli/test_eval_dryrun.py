import json

from omegaconf import OmegaConf

from hippo_eval.bench import run_suite, write_outputs


def test_eval_dryrun(tmp_path) -> None:
    cfg = OmegaConf.create({"preset": "baselines/core", "suite": "episodic_cross_mem", "n": 2, "seed": 0})
    run, flat_ablate = run_suite(cfg)
    write_outputs(tmp_path, run, flat_ablate, cfg)
    data = json.loads((tmp_path / "metrics.json").read_text())
    assert data["metrics"]["episodic_cross_mem"]["em_raw"] >= 0.0
