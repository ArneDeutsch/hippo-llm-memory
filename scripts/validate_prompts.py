# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Validate that test prompts do not contain leaked facts.

The script checks that no token from the teach prompts appears in the
corresponding test prompts.  It exits with status 1 when a leak is
detected and prints the offending token and line number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teach", type=Path, help="Path to teach JSONL file")
    parser.add_argument("test", type=Path, help="Path to test JSONL file")
    return parser.parse_args()


def _load_tokens(path: Path) -> set[str]:
    """Return a set of whitespace tokens from ``path``."""

    tokens: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt") or "")
            tokens.update(prompt.split())
    return tokens


def _check_leak(teach_tokens: set[str], test_path: Path) -> None:
    """Raise ``ValueError`` if a leaked token is found in test prompts."""

    with test_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt") or "")
            for tok in teach_tokens:
                if tok and tok in prompt:
                    raise ValueError(f"leaked token '{tok}' in test line {idx}")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    try:
        tokens = _load_tokens(args.teach)
        _check_leak(tokens, args.test)
    except Exception as err:  # pragma: no cover - CLI behaviour
        print(err, file=sys.stderr)
        raise SystemExit(1) from err


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
