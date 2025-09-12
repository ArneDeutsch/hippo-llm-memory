# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from omegaconf import DictConfig, OmegaConf

from .layout import bench_paths

if TYPE_CHECKING:
    from . import BenchRun


@dataclass
class BenchResult:
    """Container for a single bench run."""

    outdir: Path
    run: BenchRun
    flat_ablate: Dict[str, object]


def run_bench(cfg: DictConfig) -> BenchResult:
    """Execute one bench run and persist metrics."""
    from . import run_suite, write_outputs

    paths = bench_paths(cfg)
    run, flat_ablate = run_suite(cfg)
    write_outputs(paths.outdir, run, flat_ablate, cfg)
    return BenchResult(paths.outdir, run, flat_ablate)


def run_matrix(matrix_cfg: DictConfig) -> List[BenchResult]:
    """Evaluate a grid of suites, sample counts and seeds."""

    paths = bench_paths(matrix_cfg)
    suites = matrix_cfg.get("suites", ["episodic", "semantic", "spatial"])
    n_values = matrix_cfg.get("n_values", [50, 200, 1000])
    seeds = matrix_cfg.get("seeds", [1337, 2025, 4242])
    base_cfg = OmegaConf.to_container(matrix_cfg, resolve=True)
    results: List[BenchResult] = []
    for suite in suites:
        for n in n_values:
            for seed in seeds:
                run_cfg = OmegaConf.create(base_cfg)
                run_cfg.suite = suite
                run_cfg.n = int(n)
                run_cfg.seed = int(seed)
                outdir = paths.root / suite / f"n{n}_seed{seed}"
                run_cfg.outdir = str(outdir)
                mem = run_cfg.get("memory")
                if isinstance(mem, DictConfig):
                    rel_gate = (
                        mem.relational.gate
                        if mem.get("relational") and mem.relational.get("gate")
                        else None
                    )
                    spat_gate = (
                        mem.spatial.gate if mem.get("spatial") and mem.spatial.get("gate") else None
                    )
                else:
                    rel_gate = spat_gate = None
                if rel_gate or spat_gate:
                    for enabled in [True, False]:
                        gate_cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
                        if rel_gate:
                            OmegaConf.update(
                                gate_cfg, "memory.relational.gate.enabled", enabled, merge=True
                            )
                        if spat_gate:
                            OmegaConf.update(
                                gate_cfg, "memory.spatial.gate.enabled", enabled, merge=True
                            )
                        gate_dir = outdir / ("gate_on" if enabled else "gate_off")
                        gate_cfg.outdir = str(gate_dir)
                        results.append(run_bench(gate_cfg))
                else:
                    results.append(run_bench(run_cfg))
    return results


__all__ = ["BenchResult", "run_bench", "run_matrix"]
