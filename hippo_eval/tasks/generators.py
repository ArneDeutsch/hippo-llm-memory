# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

import random
from typing import Dict, List, Union


def generate_episodic_cross_mem(
    size: int,
    seed: int,
    entity_pool: int | None = None,
    distractors: int | None = None,
    profile: str = "default",
) -> Dict[str, List[Dict[str, object]]]:
    """Generate cross-session episodes split into teach and test sets."""
    rng = random.Random(seed)
    if entity_pool is None:
        entity_pool = {"easy": 8, "default": 12, "hard": 16}[profile]
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
    teach: List[Dict[str, object]] = []
    test: List[Dict[str, object]] = []
    while len(test) < size:
        who = rng.choice(people)
        where = rng.choice(places)
        verb = rng.choice(verbs)
        when = rng.choice(times)
        key = (who, where, verb, when)
        if key in seen:
            continue
        seen.add(key)
        context_key = f"epx/{len(test):05d}"
        fact = f"{who} {verb} to the {where} {when}."
        teach.append({"fact": fact, "context_key": context_key})
        prompt = f"Where did {who} go? Answer with the location name only."
        test.append({"prompt": prompt, "answer": where, "context_key": context_key})
    return {"teach": teach, "test": test}


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
) -> Union[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
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

    if require_memory:
        teach: List[Dict[str, object]] = []
        test: List[Dict[str, object]] = []
    else:
        tasks: List[Dict[str, object]] = []
    for i in range(size):
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
        if require_memory:
            context_key = f"sem/{i:05d}"
            for fact in facts:
                teach.append(
                    {
                        "fact": fact["text"],
                        "schema_fit": fact["schema_fit"],
                        "time": fact["time"],
                        "context_key": context_key,
                    }
                )
            test.append({"prompt": prompt, "answer": city, "context_key": context_key})
        else:
            tasks.append({"prompt": prompt, "answer": city, "facts": facts})

    if require_memory:
        return {"teach": teach, "test": test}
    return tasks


__all__ = ["generate_episodic_cross_mem", "generate_semantic"]
