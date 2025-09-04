"""Render baseline metrics as Markdown table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_baselines(metrics_csv: Path, out_path: Path) -> Path:
    """Render ``metrics_csv`` into ``out_path`` using the default templates."""

    with metrics_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["md"]),
    )
    template = env.get_template("partials/baselines.md.j2")
    out_path.write_text(template.render(rows=rows), encoding="utf-8")
    return out_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    render_baselines(args.metrics, args.out)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
