import re
from statistics import mean

from hippo_eval.datasets import generate_episodic, generate_semantic


def _naive_episodic_baseline(prompt: str) -> str:
    story, question = prompt.rsplit("?", 1)
    story = story.rsplit(". ", 1)[0]
    sentences = [s.strip() for s in story.split(".") if s.strip()]
    last = sentences[-1]
    m = re.match(r"([A-Za-z]+) ([a-z]+) at the ([A-Za-z]+) on ([A-Za-z]+)", last)
    if not m:
        return ""
    who, what, where, when = m.groups()
    if "Who was at the" in question:
        return who
    if "What did" in question:
        return what
    if "Where was" in question:
        return f"the {where}"
    return when


def _count_sentences(prompt: str) -> int:
    story = prompt.rsplit(". ", 1)[0]
    return len([s for s in story.split(".") if s.strip()])


def test_episodic_profile_affects_distractors_and_accuracy():
    easy = generate_episodic(20, 0, profile="easy")
    hard = generate_episodic(20, 0, profile="hard")
    # easy prompts contain only the target sentence
    assert all(_count_sentences(p["prompt"]) == 1 for p in easy)
    # hard prompts have extra sentences
    assert any(_count_sentences(p["prompt"]) > 1 for p in hard)
    easy_em = mean(_naive_episodic_baseline(t["prompt"]) == t["answer"] for t in easy)
    hard_em = mean(_naive_episodic_baseline(t["prompt"]) == t["answer"] for t in hard)
    assert easy_em > hard_em


def test_semantic_profile_injects_contradictions():
    easy = generate_semantic(5, 0, profile="easy")
    hard = generate_semantic(5, 0, profile="hard")
    assert not any("However" in item["prompt"] for item in easy)
    assert any("However" in item["prompt"] for item in hard)


def test_semantic_hard_profile_adds_distractors_and_entity_overlap():
    easy = generate_semantic(5, 0, profile="easy")
    hard = generate_semantic(5, 0, profile="hard")
    # hard prompts contain extra distractor sentences
    assert max(_count_sentences(t["prompt"]) for t in hard) > max(
        _count_sentences(t["prompt"]) for t in easy
    )
    store_re = re.compile(r"at (Store[A-Z])")
    easy_stores = {store_re.search(t["prompt"]).group(1) for t in easy}
    hard_stores = {store_re.search(t["prompt"]).group(1) for t in hard}
    # entity pool restriction reduces variety of store names
    assert len(hard_stores) <= len(easy_stores)
