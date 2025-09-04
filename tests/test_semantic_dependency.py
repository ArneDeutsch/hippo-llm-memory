"""Ensure semantic suite answers depend on the knowledge graph."""

from __future__ import annotations

from hippo_eval.tasks.generators import generate_semantic
from hippo_mem.memory import evaluate_semantic


def test_semantic_requires_kg() -> None:
    """Answering without the KG drops exact match by at least 0.2."""

    data = generate_semantic(size=20, seed=0, require_memory=True)
    em_with = evaluate_semantic(data, use_kg=True)
    em_without = evaluate_semantic(data, use_kg=False)
    assert em_with >= 0.8
    assert em_with - em_without >= 0.2
