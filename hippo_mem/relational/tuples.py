"""Utilities for extracting simple relational tuples from text.

The goal of this module is not to be a full information extraction system but
rather to provide a very small, easy to understand starting point. The
``extract_tuples`` function implements a handful of heuristic rules which treat
each sentence in the input text as describing a simple ``head -> relation ->
tail`` triple.  A four digit year, if present, is interpreted as the temporal
marker.  The original sentence is kept as the ``context`` field and the sentence
index is used as provenance.

This simplistic representation is sufficient for unit tests and for building a
toy knowledge graph in :mod:`hippo_mem.relational.kg`.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

TupleType = Tuple[str, str, str, str, Optional[str], float, int]


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences using basic punctuation."""

    return [s for s in re.split(r"[.?!]\s*", text.strip()) if s]


def strip_time(sent: str) -> tuple[str, Optional[str]]:
    """Remove temporal markers like ``in 2020`` and return ``(sent, year)``."""

    time_match = re.search(r"\b(\d{4})\b", sent)
    if not time_match:
        return sent, None
    time = time_match.group(1)
    sent_no_time = re.sub(
        rf"\b(?:in|on|at)\s*{time}\b",
        "",
        sent,
        flags=re.IGNORECASE,
    ).strip()
    return sent_no_time, time


def _parse_triplet(sent: str) -> Tuple[str, str, str]:
    """Very small heuristic parser returning ``(head, relation, tail)``."""

    tokens = sent.split()
    if len(tokens) < 3:
        return tokens[0], "", " ".join(tokens[1:]) if len(tokens) > 1 else ""
    head = tokens[0].strip()
    relation = tokens[1].strip()
    tail = " ".join(tokens[2:]).strip()
    return head, relation, tail


def score_confidence(relation: str, tail: str, time: Optional[str] = None) -> float:
    """Heuristic confidence score from relation/tail length and time."""

    rel_len = len(relation.split()) + len(tail.split())
    conf = min(1.0, rel_len / 3.0)
    if time is not None:
        conf = min(1.0, conf + 0.1)
    return conf


def extract_tuples(text: str, threshold: float = 0.0) -> List[TupleType]:
    """Extract ``(head, relation, tail, context, time, conf, provenance)`` tuples."""

    tuples: List[TupleType] = []
    for idx, sent in enumerate(split_sentences(text)):
        sent_no_time, time = strip_time(sent)
        head, relation, tail = _parse_triplet(sent_no_time)
        if not relation or not tail:
            continue

        conf = score_confidence(relation, tail, time)
        if conf < threshold:
            continue

        tuples.append((head, relation, tail, sent.strip(), time, conf, idx))

    return tuples


__all__ = [
    "extract_tuples",
    "TupleType",
    "split_sentences",
    "strip_time",
    "score_confidence",
]
