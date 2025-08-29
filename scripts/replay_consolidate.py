"""CLI wrapper for :mod:`hippo_mem.consolidation.trainer`.

The script exists as a thin entry point so it can be executed without
installing the package.  All implementation lives in the
``hippo_mem.consolidation.trainer`` module.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch - idempotent
    sys.path.insert(0, str(ROOT))

from hippo_mem.consolidation.trainer import (  # noqa: E402
    Args,
    load_config,
    parse_args,
    train,
)
from hippo_mem.utils.stores import assert_store_exists  # noqa: E402

__all__ = ["Args", "load_config", "main", "parse_args", "train"]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    assert_store_exists(args.store_dir, args.session_id, kind="episodic")
    args.store_dir = str(Path(args.store_dir) / "hei_nw")
    cfg = load_config(args.config)
    train(args, cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
