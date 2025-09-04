from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hippo_eval.bench import run_suite, write_outputs


@pytest.mark.smoke
def test_end_to_end_smoke(tmp_path: Path) -> None:
    """Run baseline and memory presets in-process and validate metrics."""

    # Baseline preset
    base_cfg = OmegaConf.create(
        {"preset": "baselines/core", "suite": "episodic", "n": 2, "seed": 1337}
    )
    rows, metrics, ablate = run_suite(base_cfg)
    baseline_out = tmp_path / "baseline"
    write_outputs(baseline_out, rows, metrics, ablate, base_cfg)
    assert (baseline_out / "metrics.json").exists()

    # Memory preset
    mem_cfg = OmegaConf.create(
        {
            "preset": "memory/hei_nw",
            "suite": "episodic",
            "n": 2,
            "seed": 1337,
            "memory": "hei_nw",
        }
    )
    rows_mem, metrics_mem, ablate_mem = run_suite(mem_cfg)
    mem_out = tmp_path / "memory"
    write_outputs(mem_out, rows_mem, metrics_mem, ablate_mem, mem_cfg)
    assert (mem_out / "metrics.json").exists()

    # Basic sanity: EM metrics present
    assert "episodic" in metrics["metrics"]
    assert "episodic" in metrics_mem["metrics"]
