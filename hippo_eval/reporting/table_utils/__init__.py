# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Utilities for rendering report tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader

_DISPLAY_NAMES = {
    "em_raw": "EM (raw)",
    "em_norm": (
        '<span title="Normalized exact match (lowercase, no punctuation or '
        'articles)">EM (norm)</span>'
    ),
    "em": "EM",
    "pre_em": "EM (pre)",
    "post_em": "EM (post)",
    "delta_em": "ΔEM",
    "pre_em_norm": "EM (pre, norm)",
    "post_em_norm": "EM (post, norm)",
    "delta_em_norm": "ΔEM (norm)",
    "pre_f1": "F1 (pre)",
    "post_f1": "F1 (post)",
    "delta_f1": "ΔF1",
    "overlong": "overlong",
    "format_violation": "format_violation",
    "memory_hit_rate": "memory_hit_rate",
    "latency_ms_delta": "latency_ms_delta",
    "context_match_rate": "context_match_rate",
    "justification_coverage": "justification_coverage",
    "gate_accept_rate": "gate_accept_rate",
    "uplift_vs_longctx_em": "ΔEM vs longctx",
    "uplift_vs_longctx_f1": "ΔF1 vs longctx",
}

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
ENV = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def display_name(key: str) -> str:
    """Return a human-friendly display name for ``key``."""

    return _DISPLAY_NAMES.get(key, key)


def format_stat(stat: tuple[float, float]) -> str:
    """Format a ``(mean, ci)`` tuple with 3 decimals."""

    mean_val, ci = stat
    return f"{mean_val:.3f} ± {ci:.3f}"


def _normalize(df: object) -> tuple[list[Mapping[str, object]], Sequence[str]]:
    """Return records and columns for ``df`` supporting DataFrame-like objects."""

    if hasattr(df, "to_dict") and hasattr(df, "columns"):
        records = df.to_dict(orient="records")  # type: ignore[arg-type]
        columns = list(df.columns)  # type: ignore[attr-defined]
        return records, columns
    if isinstance(df, Iterable):
        records = [dict(r) for r in df]  # type: ignore[misc]
        columns = list(records[0].keys()) if records else []
        return records, columns
    raise TypeError("df must be a DataFrame or iterable of mappings")


def _make_table(records: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for rec in records:
        row = [str(rec.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_metrics_table(df: object) -> str:
    """Return a Markdown metrics table for DataFrame ``df``."""

    records, columns = _normalize(df)
    return _make_table(records, columns)


def make_summary_table(df: object) -> str:
    """Return a Markdown summary table for DataFrame ``df``."""

    records, columns = _normalize(df)
    return _make_table(records, columns)


__all__ = [
    "TEMPLATE_DIR",
    "ENV",
    "display_name",
    "format_stat",
    "make_metrics_table",
    "make_summary_table",
]
