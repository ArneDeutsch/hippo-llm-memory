"""Synthetic dataset generators for evaluation suites."""

from __future__ import annotations

import random
from typing import Dict, List

from .spatial.generator import generate_spatial

_CAPACITY_FILLER_SENTENCES = [
    "The sky is blue.",
    "Birds sing at dawn.",
    "She closed the door.",
    "Rain pattered softly.",
    "A cat napped nearby.",
    "Wind rustled the leaves.",
    "They walked along quietly.",
    "Stars twinkled above.",
]


def generate_episodic(
    size: int,
    seed: int,
    distractors: int | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Generate ``size`` W4 stories with optional distractors and swaps."""
    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"]
    actions = ["met", "saw", "helped", "found"]
    places = ["Cafe", "Library", "Park", "Mall"]
    times = ["Monday", "Tuesday", "Wednesday", "Thursday"]
    qtypes = ["who_at_where", "what_did_who", "where_was_who", "when_was_who"]

    if distractors is None:
        distractors = {"easy": 0, "default": 2, "hard": 4}[profile]
    post_distractors = 1 if profile == "hard" else 0

    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        qtype = rng.choice(qtypes)

        who = rng.choice(people)
        what = rng.choice(actions)
        where = rng.choice(places)
        when = rng.choice(times)
        target_sent = f"{who} {what} at the {where} on {when}."

        pre_sents: List[str] = []
        for _ in range(distractors):
            candidates = people if qtype == "who_at_where" else [p for p in people if p != who]
            dwho = rng.choice(candidates)
            dwhat = rng.choice(actions)
            dwhere = rng.choice(places)
            dwhen = rng.choice(times)
            pre_sents.append(f"{dwho} {dwhat} at the {dwhere} on {dwhen}.")

        post_sents: List[str] = []
        for _ in range(post_distractors):
            pdwho = rng.choice([p for p in people if p != who])
            pdwhat = rng.choice(actions)
            pdwhen = rng.choice(times)
            post_sents.append(f"{pdwho} {pdwhat} at the {where} on {pdwhen}.")

        if qtype == "who_at_where":
            question = f"Who was at the {where}?"
            answer = who
        elif qtype == "what_did_who":
            question = f"What did {who} do?"
            answer = what
        elif qtype == "where_was_who":
            question = f"Where was {who}?"
            answer = f"the {where}"
        else:  # when_was_who
            question = f"When was {who} at the {where}?"
            answer = when

        story = " ".join(pre_sents + [target_sent] + post_sents)
        reward = rng.random() < 0.3
        pin = rng.random() < 0.1
        tasks.append(
            {
                "prompt": f"{story} {question}",
                "answer": answer,
                "reward": reward,
                "pin": pin,
            }
        )
    return tasks


def generate_episodic_multi(
    size: int,
    seed: int,
    distractors: int | None = None,
    max_corrections: int | None = None,
    omit_fraction: float = 0.0,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Multi-turn episodes with distractors and optional corrections."""
    if distractors is None:
        distractors = {"easy": 8, "default": 10, "hard": 12}[profile]
    if max_corrections is None:
        max_corrections = 2

    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"]
    colors = ["red", "blue", "green", "yellow"]
    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        who = rng.choice(people)
        first_color = rng.choice(colors)
        distractor_sents = [
            f"Distractor: {rng.choice(people)} likes {rng.choice(colors)}."
            for _ in range(distractors)
        ]
        sents = distractor_sents + [f"{who} likes {first_color}."]

        num_corrections = 0
        if max_corrections > 0 and rng.random() >= omit_fraction:
            num_corrections = rng.randint(0, max_corrections)

        current_color = first_color
        for _ in range(num_corrections):
            new_color = rng.choice([c for c in colors if c != current_color])
            sents.append(f"Actually, {who} likes {new_color}.")
            current_color = new_color

        answer = current_color
        question = f"What color does {who} like?"
        prompt = " ".join(sents + [question])
        tasks.append({"prompt": prompt, "answer": answer})
    return tasks


def generate_episodic_cross(
    size: int,
    seed: int,
    entity_pool: int | None = None,
    distractors: int | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Cross-episode recall after a flush marker."""
    rng = random.Random(seed)
    if entity_pool is None:
        entity_pool = {"easy": 8, "default": 12, "hard": 16}[profile]
    if distractors is None:
        distractors = {"easy": 1, "default": 2, "hard": 4}[profile]
    base_people = [
        "Alice",
        "Bob",
        "Carol",
        "Dave",
        "Eve",
        "Frank",
        "Grace",
        "Heidi",
        "Ivan",
        "Judy",
        "Mallory",
        "Niaj",
        "Olivia",
        "Peggy",
        "Rupert",
        "Sybil",
    ]
    base_places = [
        "Cafe",
        "Library",
        "Park",
        "Mall",
        "Office",
        "School",
        "Cinema",
        "Museum",
        "Beach",
        "Zoo",
        "Stadium",
        "Theater",
        "Bank",
        "Restaurant",
        "Hospital",
        "Station",
    ]
    verbs = ["went", "traveled", "walked", "drove", "journeyed"]
    times = [
        "yesterday",
        "last week",
        "this morning",
        "earlier today",
        "in the evening",
    ]
    people = base_people[:entity_pool]
    places = base_places[:entity_pool]
    combinations = len(people) * len(places) * len(verbs) * len(times)
    if size > combinations:
        raise ValueError("size exceeds unique episode capacity")
    seen: set[tuple[str, str, str, str]] = set()
    tasks: List[Dict[str, object]] = []
    while len(tasks) < size:
        who = rng.choice(people)
        where = rng.choice(places)
        verb = rng.choice(verbs)
        when = rng.choice(times)
        key = (who, where, verb, when)
        if key in seen:
            continue
        seen.add(key)
        fact = f"{who} {verb} to the {where} {when}."
        post_sents: List[str] = []
        for _ in range(distractors):
            dwho = rng.choice([p for p in people if p != who])
            dwhere = rng.choice([pl for pl in places if pl != where])
            dverb = rng.choice(verbs)
            dwhen = rng.choice(times)
            post_sents.append(f"{dwho} {dverb} to the {dwhere} {dwhen}.")
        distractor_text = " ".join(post_sents)
        prompt = (
            f"{fact} --- FLUSH --- {distractor_text} Where did {who} go? "
            "Answer with the location name only."
        )
        tasks.append({"prompt": prompt, "answer": where})
    return tasks


def generate_episodic_capacity(
    size: int,
    seed: int,
    context_budget: int | None = None,
    profile: str = "default",
    target_length: int | None = None,
) -> List[Dict[str, object]]:
    """Episodes exceeding the decoding context budget."""
    rng = random.Random(seed)
    if context_budget is None:
        context_budget = {"easy": 256, "default": 384, "hard": 512}[profile]
    people = ["Alice", "Bob", "Carol", "Dave"]
    places = ["Cafe", "Library", "Park", "Mall"]
    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        who = rng.choice(people)
        where = rng.choice(places)
        fact = f"{who} went to the {where}."
        question = f"Where did {who} go?"
        base_tokens = len(fact.split()) + len(question.split())
        if target_length is None:
            target = context_budget + base_tokens + 10
        else:
            target = target_length
        filler_tokens_needed = max(target - base_tokens, 0)
        filler_tokens: List[str] = []
        while len(filler_tokens) < filler_tokens_needed:
            filler_tokens.extend(rng.choice(_CAPACITY_FILLER_SENTENCES).split())
        filler_text = " ".join(filler_tokens[:filler_tokens_needed])
        prompt = f"{fact} {filler_text} {question}".strip()
        tasks.append({"prompt": prompt, "answer": f"the {where}"})
    return tasks


def generate_semantic(
    size: int,
    seed: int,
    hop_depth: int | None = None,
    inject_contradictions: bool | None = None,
    require_memory: bool = False,
    distractors: int | None = None,
    entity_pool: int | None = None,
    paraphrase_prob: float | None = None,
    ambiguity_prob: float | None = None,
    profile: str = "default",
) -> List[Dict[str, object]]:
    """Generate 2–3 hop fact chains linking people, items, stores and cities."""
    if hop_depth is None:
        hop_depth = 2 if profile != "hard" else 3
    if inject_contradictions is None:
        inject_contradictions = profile in {"default", "hard"}
    if distractors is None:
        distractors = 0 if profile != "hard" else 2
    if entity_pool is None:
        entity_pool = 4 if profile != "hard" else 2
    if paraphrase_prob is None:
        paraphrase_prob = 0.0 if profile != "hard" else 0.3
    if ambiguity_prob is None:
        ambiguity_prob = 0.0 if profile != "hard" else 0.3
    if hop_depth not in {2, 3}:
        raise ValueError("hop_depth must be 2 or 3")

    rng = random.Random(seed)
    people = ["Alice", "Bob", "Carol", "Dave"][:entity_pool]
    items = ["book", "apple", "ball", "coin"][:entity_pool]
    stores = ["StoreA", "StoreB", "StoreC"][:entity_pool]
    cities = ["Paris", "London", "Rome", "Berlin"][:entity_pool]

    def _buy_verb() -> str:
        verbs = ["bought", "purchased"]
        return rng.choice(verbs) if rng.random() < paraphrase_prob else verbs[0]

    def _sold_phrase() -> str:
        phrases = ["was sold at", "could be found at"]
        return rng.choice(phrases) if rng.random() < paraphrase_prob else phrases[0]

    def _is_in_phrase() -> str:
        phrases = ["is in", "is located in"]
        return rng.choice(phrases) if rng.random() < paraphrase_prob else phrases[0]

    tasks: List[Dict[str, object]] = []
    for _ in range(size):
        who = rng.choice(people)
        item = rng.choice(items)
        store = rng.choice(stores)
        city = rng.choice(cities)

        parts: List[str] = []
        facts: List[Dict[str, object]] = []
        for _ in range(distractors):
            dwho = rng.choice(people)
            ditem = rng.choice(items)
            dstore = rng.choice(stores)
            dcity = rng.choice(cities)
            sent = f"{dwho} {_buy_verb()} a {ditem} at {dstore}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": True, "time": len(facts)})
            is_in = _is_in_phrase()
            sent = f"{dstore} {is_in} {dcity}."
            if rng.random() < ambiguity_prob:
                sent = f"It {is_in} {dcity}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": True, "time": len(facts)})
        if hop_depth == 2:
            sent = f"{who} {_buy_verb()} a {item} at {store}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": True, "time": len(facts)})
        else:  # hop_depth == 3
            sent = f"{who} {_buy_verb()} a {item}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": True, "time": len(facts)})
            sold = _sold_phrase()
            sent = f"The {item} {sold} {store}."
            if rng.random() < ambiguity_prob:
                sent = f"It {sold} {store}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": True, "time": len(facts)})
        is_in = _is_in_phrase()
        sent = f"{store} {is_in} {city}."
        if rng.random() < ambiguity_prob:
            sent = f"It {is_in} {city}."
        if not require_memory:
            parts.append(sent)
        facts.append({"text": sent, "schema_fit": True, "time": len(facts)})

        if inject_contradictions:
            false_city = rng.choice([c for c in cities if c != city])
            sent = f"However, others report {store} {is_in} {false_city}."
            if rng.random() < ambiguity_prob:
                sent = f"However, others report it {is_in} {false_city}."
            if not require_memory:
                parts.append(sent)
            facts.append({"text": sent, "schema_fit": False, "time": len(facts)})
            question = f"Despite conflicting reports, in which city did {who} buy the {item}?"
        else:
            question = f"In which city did {who} buy the {item}?"

        text = " ".join(parts).strip()
        prompt = f"{text} {question}" if text else question
        tasks.append({"prompt": prompt, "answer": city, "facts": facts})

    return tasks


__all__ = [
    "generate_episodic",
    "generate_episodic_multi",
    "generate_episodic_cross",
    "generate_episodic_capacity",
    "generate_semantic",
    "generate_spatial",
]
