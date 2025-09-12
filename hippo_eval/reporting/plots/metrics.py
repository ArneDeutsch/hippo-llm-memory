# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Metric plotting helpers for run reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Protocol

MetricDict = Dict[str, tuple[float, float]]
MetricStats = Dict[str, Dict[int, Dict[str, tuple[float, float]]]]


class PlotBackend(Protocol):
    """Strategy for configuring the plotting backend."""

    def setup(self) -> None:
        """Configure ``matplotlib`` before importing ``pyplot``."""


class DefaultBackend:
    """Use the default ``matplotlib`` backend."""

    def setup(self) -> None:  # pragma: no cover - trivial
        return


class AggBackend:
    """Use the non-interactive ``Agg`` backend for headless environments."""

    def setup(self) -> None:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg")


def _plot_metric_bars(
    metric_keys: list[str],
    presets: MetricStats,
    out_dir: Path,
    suite: str,
    plt,
) -> None:
    """Render bar charts for each metric."""

    for metric in metric_keys:
        labels: list[str] = []
        values: list[float] = []
        for preset in sorted(presets):
            for size in sorted(presets[preset]):
                labels.append(f"{preset}-{size}")
                stats = presets[preset][size].get(metric)
                values.append(stats[0] if stats else 0.0)
        plt.figure()
        plt.bar(labels, values)
        plt.ylabel(metric)
        plt.title(f"{suite} {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"{metric}.png")
        plt.close()


def _plot_uplift(
    metric_keys: list[str], presets: MetricStats, out_dir: Path, suite: str, plt
) -> None:
    """Render pre/post uplift plot if metrics are available."""

    if "pre_em" not in metric_keys or "post_em" not in metric_keys:
        return

    labels: list[str] = []
    pre_vals: list[float] = []
    pre_ci: list[float] = []
    post_vals: list[float] = []
    post_ci: list[float] = []
    for preset in sorted(presets):
        for size in sorted(presets[preset]):
            labels.append(f"{preset}-{size}")
            pre = presets[preset][size].get("pre_em")
            post = presets[preset][size].get("post_em")
            pre_vals.append(pre[0] if pre else 0.0)
            pre_ci.append(pre[1] if pre else 0.0)
            post_vals.append(post[0] if post else 0.0)
            post_ci.append(post[1] if post else 0.0)
    x = list(range(len(labels)))
    width = 0.4
    plt.figure()
    plt.bar(
        [i - width / 2 for i in x],
        pre_vals,
        width,
        yerr=pre_ci if any(pre_ci) else None,
        label="pre",
    )
    plt.bar(
        [i + width / 2 for i in x],
        post_vals,
        width,
        yerr=post_ci if any(post_ci) else None,
        label="post",
    )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("EM")
    plt.title(f"{suite} uplift")
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "uplift.png")
    plt.close()


def _plot_retrieval(retrieval: Dict[str, MetricDict] | None, out_dir: Path) -> None:
    """Plot retrieval metrics if available."""

    if retrieval is None:
        return

    try:  # pragma: no cover - optional plotting
        from hippo_eval.reporting.plots.retrieval import plot_retrieval

        plot_retrieval(retrieval, out_dir)
    except Exception as exc:  # pragma: no cover - matplotlib missing
        logging.getLogger(__name__).warning("failed to plot retrieval: %s", exc)


def render_suite_plots(
    suite: str,
    presets: MetricStats,
    out_dir: Path,
    retrieval: Dict[str, MetricDict] | None = None,
    backend: PlotBackend | None = None,
) -> None:
    """Render simple bar plots for one suite if ``matplotlib`` is available."""

    backend = backend or DefaultBackend()
    try:  # pragma: no cover - optional dependency
        backend.setup()
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib missing
        logging.getLogger(__name__).warning("matplotlib unavailable: %s", exc)
        return

    metric_keys = sorted(
        {key for preset in presets.values() for size_stats in preset.values() for key in size_stats}
    )
    _plot_metric_bars(metric_keys, presets, out_dir, suite, plt)
    _plot_uplift(metric_keys, presets, out_dir, suite, plt)
    _plot_retrieval(retrieval, out_dir)
    if suite == "spatial_multi":
        _plot_learning_curve(presets, out_dir, suite, plt)
        _write_macro_reuse_table(presets, out_dir)


def _plot_learning_curve(presets: MetricStats, out_dir: Path, suite: str, plt) -> None:
    """Plot per-episode success rates if available."""

    for preset, sizes in presets.items():
        for size, stats in sizes.items():
            curve: list[float] = []
            ep = 1
            while (key := f"success_ep{ep}") in stats:
                val = stats[key]
                curve.append(val[0] if isinstance(val, tuple) else float(val))
                ep += 1
            if curve:
                plt.figure()
                plt.plot(range(1, len(curve) + 1), curve, marker="o")
                plt.ylim(0, 1)
                plt.xlabel("Episode")
                plt.ylabel("Success rate")
                plt.title(f"{suite} {preset}-{size} learning curve")
                out_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir / f"{preset}-{size}_learning_curve.png")
                plt.close()


def _write_macro_reuse_table(presets: MetricStats, out_dir: Path) -> None:
    """Write a table with macro reuse deltas if available."""

    lines = ["| Preset | Size | Δsteps | Δlatency_ms |", "|---|---:|---:|---:|"]
    wrote = False
    for preset, sizes in presets.items():
        for size, stats in sizes.items():
            steps = stats.get("macro_reuse_plen_delta")
            lat = stats.get("macro_reuse_latency_delta_ms")
            if steps or lat:
                steps_v = steps[0] if steps else 0.0
                lat_v = lat[0] if lat else 0.0
                lines.append(f"{preset} | {size} | {steps_v:.2f} | {lat_v:.2f}")
                wrote = True
    if wrote:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "macro_reuse.md").write_text("\n".join(lines), encoding="utf-8")
