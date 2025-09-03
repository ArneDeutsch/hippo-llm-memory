from hippo_mem.eval import datasets as build_datasets


def test_generate_episodic_multi_deterministic() -> None:
    items1 = build_datasets.generate_episodic_multi(size=3, seed=0)
    items2 = build_datasets.generate_episodic_multi(size=3, seed=0)
    assert items1 == items2
    assert len(items1) == 3
    assert any("distractor" in it["prompt"].lower() for it in items1)


def test_generate_episodic_multi_varied_corrections() -> None:
    items = build_datasets.generate_episodic_multi(size=10, seed=0, max_corrections=2)
    counts = [it["prompt"].count("Actually") for it in items]
    assert any(c == 0 for c in counts)
    assert any(c >= 2 for c in counts)


def test_generate_episodic_multi_no_corrections() -> None:
    items = build_datasets.generate_episodic_multi(
        size=3, seed=0, max_corrections=2, omit_fraction=1.0
    )
    assert all("Actually" not in it["prompt"] for it in items)


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
