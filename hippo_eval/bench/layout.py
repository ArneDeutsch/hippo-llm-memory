# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def bench_paths(cfg: DictConfig) -> SimpleNamespace:
    """Return output paths for a bench run."""

    preset_path = Path(str(cfg.preset))
    run_id = cfg.run_id
    outdir_opt: Optional[str] = cfg.get("outdir")
    if outdir_opt is not None:
        outdir_path = Path(to_absolute_path(outdir_opt))
        root_outdir = outdir_path if cfg.get("run_matrix") else outdir_path.parent.parent
    else:
        if preset_path.parts and preset_path.parts[0] == "baselines":
            root_outdir = Path("runs") / run_id / preset_path.parts[0] / preset_path.name
        else:
            root_outdir = Path("runs") / run_id / preset_path.name
        if cfg.get("run_matrix"):
            outdir_path = root_outdir
        else:
            outdir_path = root_outdir / cfg.suite
    return SimpleNamespace(root=root_outdir, outdir=outdir_path)


__all__ = ["bench_paths"]
