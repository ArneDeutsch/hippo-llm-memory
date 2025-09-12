# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Ensure semantic suite answers depend on the knowledge graph."""

from __future__ import annotations

from hippo_eval.datasets import generate_semantic
from hippo_mem.memory import evaluate_semantic


def test_semantic_requires_kg() -> None:
    """Answering without the KG drops exact match by at least 0.2."""

    raw = generate_semantic(size=20, seed=0, require_memory=True)
    items = []
    for test_item in raw["test"]:
        facts = [t for t in raw["teach"] if t["context_key"] == test_item["context_key"]]
        items.append(
            {
                "prompt": test_item["prompt"],
                "answer": test_item["answer"],
                "facts": [{"text": f["fact"], "schema_fit": f["schema_fit"]} for f in facts],
            }
        )
    em_with = evaluate_semantic(items, use_kg=True)
    em_without = evaluate_semantic(items, use_kg=False)
    assert em_with >= 0.8
    assert em_with - em_without >= 0.2
