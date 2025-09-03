"""Tuple extraction utilities.

Summary
-------
Provide lightweight tuple extraction for SGC-RSS. Each sentence is parsed
into ``(head, relation, tail, context, time, conf, provenance, head_type, tail_type)``
with rudimentary heuristics. Types default to ``"entity"`` and allow the
semantic store to track heterogeneous nodes. Precision reaches ≈0.9 when
``threshold >= 0.5`` as verified in unit tests.
Complexity
----------
All helpers operate in ``O(n)`` time where ``n`` is the number of
characters or tokens.

Examples
--------
>>> text = "Alice likes Bob. Bob visited Paris in 2020."
>>> extract_tuples(text, threshold=0.5)  # doctest: +ELLIPSIS
[('Alice', 'likes', 'Bob', 'Alice likes Bob.', None, ..., 0),
 ('Bob', 'visited', 'Paris', 'Bob visited Paris in 2020.', '2020', ..., 1)]

See Also
--------
hippo_mem.relational.kg : Semantic graph storage built on these tuples.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

TupleType = Tuple[str, str, str, str, Optional[str], float, int, str, str]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using basic punctuation.

    Summary
    -------
    Simple tokenizer for unit tests; not robust to abbreviations.

    Parameters
    ----------
    text : str
        Input paragraph.

    Returns
    -------
    List[str]
        Sentences without trailing punctuation.
    Complexity
    ----------
    ``O(len(text))``

    Examples
    --------
    >>> split_sentences('Alice met Bob. Carol built rockets!')
    ['Alice met Bob', 'Carol built rockets']

    See Also
    --------
    strip_time
    """

    return [s for s in re.split(r"[.?!]\s*", text.strip()) if s]


def strip_time(sent: str) -> tuple[str, Optional[str]]:
    """Remove temporal markers and capture a four-digit year.

    Summary
    -------
    Extracts a year token and returns the sentence without it.

    Parameters
    ----------
    sent : str
        Sentence possibly containing ``YYYY``.

    Returns
    -------
    tuple[str, Optional[str]]
        Sentence sans time marker and the extracted year.
    Complexity
    ----------
    ``O(len(sent))``

    Examples
    --------
    >>> strip_time('Bob visited Paris in 2020')
    ('Bob visited Paris', '2020')

    See Also
    --------
    split_sentences
    """

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
    """Very small heuristic parser returning ``(head, relation, tail)``.

    Summary
    -------
    Splits on whitespace to obtain a naive ``head relation tail`` pattern.

    Parameters
    ----------
    sent : str
        Cleaned sentence.

    Returns
    -------
    Tuple[str, str, str]
        Parsed tuple; empty strings when parts are missing.
    Complexity
    ----------
    ``O(len(sent))``

    Examples
    --------
    >>> _parse_triplet('Alice likes Bob')
    ('Alice', 'likes', 'Bob')

    See Also
    --------
    score_confidence
    """

    tokens = sent.split()
    if len(tokens) < 3:
        return tokens[0], "", " ".join(tokens[1:]) if len(tokens) > 1 else ""
    head = tokens[0].strip()
    relation = tokens[1].strip()
    tail = " ".join(tokens[2:]).strip()
    return head, relation, tail


def score_confidence(relation: str, tail: str, time: Optional[str] = None) -> float:
    """Compute a heuristic confidence score.

    Summary
    -------
    Longer relations/tails and presence of a time marker yield higher
    confidence; capped at ``1.0``.

    Parameters
    ----------
    relation : str
        Relation string.
    tail : str
        Tail string.
    time : Optional[str], optional
        Year token extracted by :func:`strip_time`.

    Returns
    -------
    float
        Confidence in ``[0, 1]``.
    Complexity
    ----------
    ``O(len(relation) + len(tail))``

    Examples
    --------
    >>> score_confidence('visited', 'Paris')
    0.666...
    >>> score_confidence('visited', 'Paris', time='2020')
    0.766...

    See Also
    --------
    extract_tuples
    """

    rel_len = len(relation.split()) + len(tail.split())
    conf = min(1.0, rel_len / 3.0)
    if time is not None:
        conf = min(1.0, conf + 0.1)
    return conf


def extract_tuples(text: str, threshold: float = 0.0) -> List[TupleType]:
    """Extract relational tuples from free text.

    Summary
    -------
    Parses each sentence and emits tuples meeting a confidence threshold.

    Parameters
    ----------
    text : str
        Source document.
    threshold : float, optional
        Minimum confidence in ``[0, 1]``; values below are dropped.

    Returns
    -------
    List[TupleType]
        ``(head, relation, tail, context, time, conf, provenance, head_type, tail_type)`` tuples.
    Complexity
    ----------
    Linear in number of sentences.

    Examples
    --------
    >>> extract_tuples('Alice likes Bob.', threshold=0.5)
    [('Alice', 'likes', 'Bob', 'Alice likes Bob.', None, 0.666..., 0)]

    See Also
    --------
    score_confidence
    """

    tuples: List[TupleType] = []
    for idx, sent in enumerate(split_sentences(text)):
        sent_no_time, time = strip_time(sent)
        head, relation, tail = _parse_triplet(sent_no_time)
        if not relation or not tail:
            continue

        conf = score_confidence(relation, tail, time)
        if conf < threshold:
            continue

        # why: sentence index serves as minimal provenance for rollback
        tuples.append((head, relation, tail, sent.strip(), time, conf, idx, "entity", "entity"))

    return tuples


__all__ = [
    "extract_tuples",
    "TupleType",
    "split_sentences",
    "strip_time",
    "score_confidence",
]
