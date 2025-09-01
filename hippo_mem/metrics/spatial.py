from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

from hippo_mem.metrics.scoring import _MOVE_DIRS, _parse_coord, _parse_moves

_WORD_TO_MOVE = {
    "UP": "U",
    "DOWN": "D",
    "LEFT": "L",
    "RIGHT": "R",
}

_COORD_RE = re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)|\[\s*-?\d+\s*,\s*-?\d+\s*\]|-?\d+\s*,\s*-?\d+")


def ensure_prediction_format(text: str) -> str:
    """Return canonical move string from ``text``.

    The parser accepts coordinate traces (``(x,y)->(x,y)``), whitespace and
    comma separated moves, or words like ``up``/``left``. Unrecognised tokens
    are discarded.
    """

    text = text.strip().upper().replace("→", "->")
    for word, step in _WORD_TO_MOVE.items():
        text = re.sub(rf"\b{word}\b", step, text)

    moves = _parse_moves(text)
    if moves:
        return "".join(moves)

    coords = _COORD_RE.findall(text)
    if len(coords) >= 2:
        pts: List[Tuple[int, int]] = []
        for c in coords:
            pt = _parse_coord(c)
            if pt is None:
                pts = []
                break
            pts.append(pt)
        if len(pts) >= 2:
            steps: List[str] = []
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                dx, dy = x2 - x1, y2 - y1
                for step, (sx, sy) in _MOVE_DIRS.items():
                    if (dx, dy) == (sx, sy):
                        steps.append(step)
                        break
            if steps:
                return "".join(steps)

    return "".join(ch for ch in text if ch in _MOVE_DIRS)


def em(pred: str, gold: str) -> int:
    """Exact match after canonicalization."""

    return int(ensure_prediction_format(pred) == ensure_prediction_format(gold))


def f1(pred: str, gold: str) -> float:
    """Character-level F1 on canonical move strings."""

    pt = list(ensure_prediction_format(pred))
    gt = list(ensure_prediction_format(gold))
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)
