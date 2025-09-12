# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hippo_mem.memory import evaluate_semantic  # noqa: E402

DATA = Path(__file__).resolve().parent / "semantic.jsonl"


def main(threshold: float = 0.2) -> None:
    data = [json.loads(line) for line in DATA.read_text().splitlines()]
    em_with = evaluate_semantic(data, use_kg=True)
    em_without = evaluate_semantic(data, use_kg=False)
    delta = em_with - em_without
    if delta < threshold:
        raise SystemExit(f"EM uplift {delta:.2f} < {threshold}")
    print(f"EM uplift {delta:.2f} >= {threshold}")


if __name__ == "__main__":
    main()
