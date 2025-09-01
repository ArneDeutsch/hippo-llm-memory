"""Mark evaluation runs with stub data as invalid.

The script scans a run directory for ``store_meta.json`` files with
``"source": "stub"`` or ``metrics.json`` files where ``replay.samples`` is
zero.  If any such markers are found an ``INVALID`` file is created in the
run root and, when ``--quarantine`` is passed, the ``stores`` directory is
moved aside to ``stores_quarantine``.

The command is idempotent: re-running it will not move files again and the
presence of an existing ``INVALID`` marker short circuits the checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _run_is_invalid(run_dir: Path) -> bool:
    """Return ``True`` if ``run_dir`` contains stub stores or zero replay."""

    marker = run_dir / "INVALID"
    if marker.exists():
        return True
    for meta_path in run_dir.rglob("store_meta.json"):
        try:
            data = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("source") == "stub":
            return True
    for metrics_path in run_dir.rglob("metrics.json"):
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("replay", {}).get("samples", 0) == 0:
            return True
    return False


def mark_run_invalid(run_dir: Path, quarantine: bool = False) -> bool:
    """Mark ``run_dir`` invalid and optionally quarantine stores."""

    run_dir = run_dir.resolve()
    marker = run_dir / "INVALID"
    if not _run_is_invalid(run_dir):
        return False
    marker.touch(exist_ok=True)
    if quarantine:
        stores = run_dir / "stores"
        dest = run_dir / "stores_quarantine"
        if stores.exists():
            dest.mkdir(exist_ok=True)
            for item in stores.iterdir():
                target = dest / item.name
                if not target.exists():
                    shutil.move(str(item), target)
    return True


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory under 'runs/'")
    parser.add_argument(
        "--quarantine", action="store_true", help="move stores to stores_quarantine"
    )
    args = parser.parse_args()
    mark_run_invalid(args.run, args.quarantine)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
