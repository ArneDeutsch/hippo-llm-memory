"""Metric plotting helpers for run reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

MetricDict = Dict[str, tuple[float, float]]
MetricStats = Dict[str, Dict[int, Dict[str, tuple[float, float]]]]


def render_suite_plots(
    suite: str,
    presets: Dict[str, Dict[int, Dict[str, tuple[float, float]]]],
    out_dir: Path,
    retrieval: Dict[str, MetricDict] | None = None,
) -> None:
    """Render simple bar plots for one suite if ``matplotlib`` is available."""

    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib missing
        logging.getLogger(__name__).warning("matplotlib unavailable: %s", exc)
        return

    metric_keys = sorted(
        {key for preset in presets.values() for size_stats in preset.values() for key in size_stats}
    )
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

    if "pre_em" in metric_keys and "post_em" in metric_keys:
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

    if retrieval:
        try:  # pragma: no cover - optional plotting
            from hippo_eval.reporting.plots.retrieval import plot_retrieval

            plot_retrieval(retrieval, out_dir)
        except Exception as exc:  # pragma: no cover - matplotlib missing
            logging.getLogger(__name__).warning("failed to plot retrieval: %s", exc)
