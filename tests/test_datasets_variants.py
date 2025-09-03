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


def test_generate_episodic_cross_single_token_answer() -> None:
    items = build_datasets.generate_episodic_cross(size=5, seed=0)
    for item in items:
        assert "location name only" in item["prompt"]
        assert len(item["answer"].split()) == 1


def test_generate_episodic_cross_inserts_distractors() -> None:
    items = build_datasets.generate_episodic_cross(size=5, seed=0)
    for item in items:
        prompt = item["prompt"]
        answer = item["answer"]
        after = prompt.split("--- FLUSH ---")[1]
        distractor = after.split("Where did")[0]
        assert distractor.strip()
        assert answer not in distractor


def test_generate_episodic_cross_unique() -> None:
    items = build_datasets.generate_episodic_cross(size=20, seed=0)
    facts = {it["prompt"].split("--- FLUSH ---")[0].strip() for it in items}
    assert len(facts) == len(items)


def test_generate_episodic_capacity_longer_than_budget() -> None:
    items1 = build_datasets.generate_episodic_capacity(size=1, seed=0, context_budget=16)
    items2 = build_datasets.generate_episodic_capacity(size=1, seed=0, context_budget=16)
    assert items1 == items2
    assert len(items1) == 1
    assert len(items1[0]["prompt"].split()) > 16


def test_generate_episodic_capacity_respects_target_length() -> None:
    target = 50
    items = build_datasets.generate_episodic_capacity(size=1, seed=0, target_length=target)
    assert len(items) == 1
    assert len(items[0]["prompt"].split()) == target
