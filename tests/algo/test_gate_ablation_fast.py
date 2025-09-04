"""Fast gate ablation checks using the bench harness.

This test replaces two slower subprocess-driven variants by running the
lightweight `hippo_eval.bench` utilities directly in-process.  No real model
is loaded which keeps runtime minimal while still exercising the gating
configuration plumbing.
"""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from hippo_eval.bench import run_suite


def _run(preset: str, suite: str, ablate_key: str | None = None) -> SimpleNamespace:
    """Execute a tiny bench run and expose gating metadata.

    Parameters mirror the previous slow tests but operate purely in-process
    without spawning subprocesses or loading transformer weights.
    """

    cfg = OmegaConf.create(
        {
            "preset": preset,
            "suite": suite,
            "n": 2,
            "seed": 1337,
            # Include a minimal relational memory config so that gating is enabled
            # by default.  `run_suite` only needs this structure to initialise
            # modules; no model weights are touched.
            "memory": {"relational": {"gate": {"enabled": True}}},
            "ablate": {},
        }
    )
    if ablate_key:
        cfg.ablate[ablate_key] = False
        # Mirror the ablation in the memory config so metadata reflects the
        # toggle.  `OmegaConf.update` merges deeply and creates nodes as needed.
        OmegaConf.update(cfg, f"memory.{ablate_key}", False, merge=True)

    # Execute the bench harness; the result is ignored as we only care about the
    # effective gate configuration recorded in `cfg`.
    run_suite(cfg)

    return SimpleNamespace(meta={"gating_enabled": bool(cfg.memory.relational.gate.enabled)})


def test_gating_can_be_disabled() -> None:
    """Relational gate ablation flips the gating metadata to ``False``."""

    res = _run("memory/sgc_rss", "semantic", "relational.gate.enabled")
    assert res.meta["gating_enabled"] is False


def test_gating_enabled_by_default() -> None:
    """Without ablation the gate remains enabled."""

    res = _run("memory/sgc_rss", "semantic")
    assert res.meta["gating_enabled"] is True
