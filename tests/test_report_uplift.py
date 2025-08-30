"""Tests for uplift reporting with confidence intervals and plots."""

from __future__ import annotations

from pathlib import Path

from scripts import report


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

    text = report._render_markdown_suite(
        "episodic", _make_presets(0.02), retrieval=None, gates=None
    )
    assert "0.500 ± 0.020" in text
    assert "saturated" in text.lower()


def test_markdown_omits_ci_with_single_seed() -> None:
    """CI column omitted for single-seed stats."""

    text = report._render_markdown_suite("episodic", _make_presets(0.0), retrieval=None, gates=None)
    assert "0.500 ±" not in text


def test_uplift_plot_written(tmp_path: Path) -> None:
    """Uplift plot is generated when pre/post metrics are present."""

    presets = _make_presets(0.02)
    report._render_plots_suite("episodic", presets, tmp_path)
    assert (tmp_path / "uplift.png").exists()
