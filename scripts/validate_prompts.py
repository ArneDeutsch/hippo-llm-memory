"""Lightweight prompt validator for memory-required test files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LEAK_PATTERNS = [" went to the ", " is in ", " was sold at ", " could be found at ", " bought "]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate test prompts contain no facts")
    parser.add_argument("path", type=Path, help="Path to <suite>_test.jsonl")
    args = parser.parse_args()
    errors: list[str] = []
    with args.path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            prompt = json.loads(line).get("prompt", "").lower()
            if any(pat in prompt for pat in LEAK_PATTERNS):
                errors.append(f"line {lineno}: leak pattern found")
    if errors:
        for msg in errors:
            print(msg)
        raise SystemExit(1)
    print(f"{args.path} OK")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
