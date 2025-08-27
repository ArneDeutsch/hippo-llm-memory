"""Scoring helpers with exact-match variants and token F1."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize(s: str) -> str:
    """Return lowercase string without punctuation or articles."""
    s = s.strip().lower()
    s = _PUNCT_RE.sub("", s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def em_raw(pred: str, gold: str) -> int:
    """Exact string match without normalization."""
    return int(pred.strip() == gold)


def em_norm(pred: str, gold: str) -> int:
    """Exact match after :func:`normalize`."""
    return int(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    """Whitespace token F1."""
    pt, gt = pred.split(), gold.split()
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)
