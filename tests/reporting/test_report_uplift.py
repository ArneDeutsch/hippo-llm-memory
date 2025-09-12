# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Tests for uplift reporting with confidence intervals and plots."""

from __future__ import annotations

from pathlib import Path

from hippo_eval.reporting.plots import AggBackend, render_suite_plots
from hippo_eval.reporting.tables import render_markdown_suite


def _make_presets(ci: float, pre_norm: float = 0.99) -> dict:
    """Return minimal preset metrics for testing."""

    stats = {
        "pre_em": (0.5, ci),
        "post_em": (0.6, ci),
        "delta_em": (0.1, ci),
        "pre_em_norm": (pre_norm, 0.0),
    }
    return {"memory": {50: stats}}


def test_markdown_includes_ci_and_saturation(tmp_path: Path) -> None:
    """Metrics table shows CI and flags saturation when pre_em_norm high."""

    text = render_markdown_suite(
        "episodic",
        _make_presets(0.02),
        retrieval=None,
        gates=None,
        seed_count=2,
    )
    assert "0.500 ± 0.020" in text
    assert "non-informative" in text.lower()


def test_markdown_includes_zero_ci_and_note_single_seed() -> None:
    """Single-seed stats show ± 0.000 and explanatory note."""

    text = render_markdown_suite(
        "episodic",
        _make_presets(0.0),
        retrieval=None,
        gates=None,
        seed_count=1,
    )
    assert "0.500 ± 0.000" in text
    assert "single-seed run: CI bands unavailable" in text


def test_uplift_plot_written(tmp_path: Path) -> None:
    """Uplift plot is generated when pre/post metrics are present."""

    presets = _make_presets(0.02)
    render_suite_plots("episodic", presets, tmp_path, backend=AggBackend())
    assert (tmp_path / "uplift.png").exists()
