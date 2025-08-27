from hippo_mem.eval import datasets as build_datasets


def test_generate_episodic_multi_deterministic() -> None:
    items1 = build_datasets.generate_episodic_multi(size=3, seed=0)
    items2 = build_datasets.generate_episodic_multi(size=3, seed=0)
    assert items1 == items2
    assert len(items1) == 3
    assert any("distractor" in it["prompt"].lower() for it in items1)


def test_generate_episodic_cross_flush() -> None:
    items1 = build_datasets.generate_episodic_cross(size=2, seed=0)
    items2 = build_datasets.generate_episodic_cross(size=2, seed=0)
    assert items1 == items2
    assert len(items1) == 2
    assert any("FLUSH" in it["prompt"] for it in items1)


def test_generate_episodic_capacity_longer_than_budget() -> None:
    items1 = build_datasets.generate_episodic_capacity(size=1, seed=0, context_budget=16)
    items2 = build_datasets.generate_episodic_capacity(size=1, seed=0, context_budget=16)
    assert items1 == items2
    assert len(items1) == 1
    assert len(items1[0]["prompt"].split()) > 16
