import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_bench


def test_meta_handles_missing_cuda_driver(tmp_path, monkeypatch):
    """``write_outputs`` records CUDA version even without driver function."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "testcuda")
    if hasattr(torch._C, "_cuda_getDriverVersion"):
        monkeypatch.delattr(torch._C, "_cuda_getDriverVersion", raising=False)

    cfg = OmegaConf.create({"seed": 0, "suite": "episodic", "preset": "baselines/core", "n": 0})
    eval_bench.write_outputs(tmp_path, [], {"acc": 1.0}, {}, cfg)

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["cuda"]["version"] == "testcuda"
    assert "driver" not in meta["cuda"]
