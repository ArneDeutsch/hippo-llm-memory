"""Utilities for teaching and querying a semantic knowledge graph.

These helpers implement a tiny knowledge graph for the synthetic
semantic suite.  During a teach phase facts are converted into triples
and written to a :class:`~hippo_mem.relational.kg.KnowledgeGraph`.
At test time prompts omit the supporting facts so answering the
question requires retrieving the city from the graph.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from hippo_mem.relational.kg import KnowledgeGraph

_PATTERN = re.compile(r"(?P<who>\w+) \w+ a (?P<item>\w+) at (?P<store>\w+).")


def _parse_facts(facts: Iterable[Dict[str, object]]) -> tuple[str, str] | None:
    """Return ``(node, city)`` pair encoded in ``facts`` or ``None``.

    ``facts`` contain a purchase sentence and a store-location sentence.
    The resulting node is ``"<person>:<item>"`` which points to the city
    where the purchase occurred.
    """

    store_line = None
    city_line = None
    for fact in facts:
        text = str(fact.get("text", ""))
        if "bought" in text:
            store_line = text
        elif " is in " in text and fact.get("schema_fit"):
            city_line = text
    if not store_line or not city_line:
        return None
    match = _PATTERN.match(store_line)
    if not match:
        return None
    who = match.group("who")
    item = match.group("item")
    store = match.group("store")
    city_match = re.match(rf"{store} .* in (?P<city>\w+).", city_line)
    if not city_match:
        return None
    city = city_match.group("city")
    node = f"{who}:{item}"
    return node, city


def teach_semantic(items: List[Dict[str, object]]) -> KnowledgeGraph:
    """Populate a :class:`KnowledgeGraph` from semantic dataset ``items``.

    Parameters
    ----------
    items:
        Dataset entries with ``facts``. Only schema-fitting facts are
        considered; contradictions are ignored.

    Returns
    -------
    KnowledgeGraph
        Graph mapping ``(person,item)`` pairs to cities.
    """

    kg = KnowledgeGraph()
    for entry in items:
        result = _parse_facts(entry.get("facts", []))
        if result is None:
            continue
        node, city = result
        context = " ".join(f["text"] for f in entry["facts"])
        kg.upsert(node, "in", city, context=context)
    return kg


def answer_question(kg: KnowledgeGraph, prompt: str) -> str:
    """Return city answer for ``prompt`` using ``kg`` or an empty string."""

    qmatch = re.search(r"which city did (\w+) buy the (\w+)", prompt)
    if not qmatch:
        return ""
    who, item = qmatch.group(1), qmatch.group(2)
    node = f"{who}:{item}"
    for _, dst, data in kg.graph.edges(node, data=True):
        if data.get("relation") == "in":
            return dst
    return ""


def evaluate_semantic(items: List[Dict[str, object]], use_kg: bool) -> float:
    """Compute exact-match accuracy with or without the knowledge graph."""

    correct = 0
    for entry in items:
        kg = teach_semantic([entry]) if use_kg else KnowledgeGraph()
        pred = answer_question(kg, entry["prompt"])
        if pred.lower() == str(entry["answer"]).lower():
            correct += 1
    return correct / len(items) if items else 0.0


__all__ = ["teach_semantic", "answer_question", "evaluate_semantic"]
