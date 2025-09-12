# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from hippo_eval import datasets as build_datasets


def test_semantic_options() -> None:
    """Semantic generator supports hop depth and contradictions."""
    three_hop = build_datasets.generate_semantic(1, seed=0, hop_depth=3)
    assert "was sold at" in three_hop[0]["prompt"]

    contradict = build_datasets.generate_semantic(1, seed=0, inject_contradictions=True)
    assert "However, others report" in contradict[0]["prompt"]


def test_semantic_require_memory_omits_facts() -> None:
    """When ``require_memory`` is set, prompts exclude fact sentences."""
    data = build_datasets.generate_semantic(1, seed=0, require_memory=True, profile="hard")
    test_item = data["test"][0]
    facts = [t["fact"] for t in data["teach"] if t["context_key"] == test_item["context_key"]]
    for fact in facts:
        assert fact not in test_item["prompt"]


def test_semantic_paraphrasing_and_pronouns() -> None:
    """Paraphrasing and pronoun ambiguity can be enabled."""
    item = build_datasets.generate_semantic(
        1,
        seed=0,
        hop_depth=3,
        paraphrase_prob=1.0,
        ambiguity_prob=1.0,
    )[0]
    prompt = item["prompt"]
    assert "It" in prompt
    assert "purchased" in prompt or "could be found at" in prompt or "is located in" in prompt


def test_semantic_three_hop_facts() -> None:
    """3-hop chains emit three schema-fit facts in order."""
    data = build_datasets.generate_semantic(1, seed=0, hop_depth=3, require_memory=True)
    test_item = data["test"][0]
    facts = [
        f["fact"]
        for f in data["teach"]
        if f["context_key"] == test_item["context_key"] and f["schema_fit"]
    ]
    assert any("bought" in f or "purchased" in f for f in facts)
    assert any("sold" in f or "found" in f for f in facts)
    assert any("is in" in f or "is located" in f for f in facts)


def test_episodic_cross_mem_split() -> None:
    """Cross-session episodic generator emits teach/test pairs."""
    data = build_datasets.generate_episodic_cross_mem(1, seed=0)
    assert len(data["teach"]) == 1
    assert len(data["test"]) == 1
    teach_item = data["teach"][0]
    test_item = data["test"][0]
    assert teach_item["context_key"] == test_item["context_key"]
    assert teach_item["fact"] not in test_item["prompt"]
