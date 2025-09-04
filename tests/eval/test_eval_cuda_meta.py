import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_bench

from hippo_eval.bench import BenchRun


@pytest.mark.parametrize("driver", [None, 123])
def test_meta_records_cuda_and_env(tmp_path, monkeypatch, driver):
    """``write_outputs`` captures CUDA metadata and other environment info."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "testcuda")
    if driver is None:
        if hasattr(torch._C, "_cuda_getDriverVersion"):
            monkeypatch.delattr(torch._C, "_cuda_getDriverVersion", raising=False)
    else:
        monkeypatch.setattr(torch._C, "_cuda_getDriverVersion", lambda: driver, raising=False)

    cfg = OmegaConf.create({"seed": 0, "suite": "episodic", "preset": "baselines/core", "n": 0})
    eval_bench.write_outputs(tmp_path, BenchRun([], {"acc": 1.0}), {}, cfg)

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["cuda"]["version"] == "testcuda"
    if driver is None:
        assert "driver" not in meta["cuda"]
    else:
        assert meta["cuda"]["driver"] == driver
    assert meta["python"] == sys.version
