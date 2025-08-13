"""Utilities for extracting simple relational tuples from text.

The goal of this module is not to be a full information extraction system but
rather to provide a very small, easy to understand starting point.  The
``extract_tuples`` function implements a handful of heuristic rules which treat
each sentence in the input text as describing a relation about the first token
("entity") in that sentence.  A four digit year, if present, is interpreted as
the temporal marker.  Everything after the first token (minus the timestamp) is
considered the relation description.  The original sentence is kept as the
``context`` field.

This simplistic representation is sufficient for unit tests and for building a
toy knowledge graph in :mod:`hippo_mem.relational.kg`.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

TupleType = Tuple[str, str, str, Optional[str]]


def extract_tuples(text: str) -> List[TupleType]:
    """Extract ``(entity, relation, context, time)`` tuples from ``text``.

    The extractor is intentionally lightweight.  It splits the text into
    sentences using common punctuation, assumes the first whitespace separated
    token of each sentence is the entity, and looks for a four digit year as a
    temporal expression.  The remainder of the sentence becomes the relation
    string and the original sentence is stored as the context.

    Args:
        text: Free form text potentially containing several sentences.

    Returns:
        A list of 4-tuples ``(entity, relation, context, time)``.  ``time`` will
        be ``None`` if no temporal expression was detected.
    """

    tuples: List[TupleType] = []
    # Split into sentences using simple punctuation rules.
    for sent in re.split(r"[.?!]\s*", text.strip()):
        if not sent:
            continue

        # Attempt to locate a four digit year.
        time_match = re.search(r"\b(\d{4})\b", sent)
        time = time_match.group(1) if time_match else None

        # Remove the time expression (and an optional preceding preposition)
        # from the sentence when constructing the relation string.
        sent_no_time = sent
        if time is not None:
            sent_no_time = re.sub(
                rf"\b(?:in|on|at)\s*{time}\b",
                "",
                sent,
                flags=re.IGNORECASE,
            ).strip()

        parts = sent_no_time.split(None, 1)
        if len(parts) < 2:
            # Not enough information for a relation.
            continue
        entity, relation = parts[0].strip(), parts[1].strip()
        tuples.append((entity, relation, sent.strip(), time))

    return tuples


__all__ = ["extract_tuples", "TupleType"]
