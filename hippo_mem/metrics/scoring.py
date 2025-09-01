"""Scoring helpers with exact-match variants and token F1."""

from __future__ import annotations

import re
import string
from collections import Counter, deque
from typing import List

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
    """Exact match after :func:`normalize`.

    ``em_norm`` is constrained so that a raw mismatch cannot yield a
    normalized match.  This prevents cases where ``em_raw`` is ``0`` but
    ``em_norm`` is ``1``.
    """
    if not em_raw(pred, gold):
        return 0
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


# -- Spatial helpers -----------------------------------------------------

_MOVE_DIRS = {
    "L": (-1, 0),
    "R": (1, 0),
    "U": (0, -1),
    "D": (0, 1),
}


def _parse_coord(text: str) -> tuple[int, int] | None:
    """Return ``(x, y)`` parsed from ``text`` if possible."""
    try:
        # ast.literal_eval handles "[x, y]" and "(x, y)" forms
        import ast

        val = ast.literal_eval(text)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return int(val[0]), int(val[1])
    except Exception:
        pass
    m = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _parse_moves(text: str) -> list[str]:
    """Return sequence of moves filtered from ``text``."""
    text = text.strip().upper()
    return [ch for ch in text if ch in _MOVE_DIRS]


def _bfs(
    start: tuple[int, int],
    goal: tuple[int, int],
    size: int,
    obstacles: set[tuple[int, int]],
) -> list[str] | None:
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
    seen = {start}
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        for step, (dx, dy) in _MOVE_DIRS.items():
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if 0 <= nx < size and 0 <= ny < size and nxt not in obstacles and nxt not in seen:
                queue.append((nxt, path + [step]))
                seen.add(nxt)
    return None


_GRID_RE = re.compile(r"Grid (\d+)x\1 with obstacles (\[[^]]*\])")
_START_RE = re.compile(r"Start (\(\d+,\s*\d+\))")
_GOAL_RE = re.compile(r"goal (\(\d+,\s*\d+\))")
_FROM_TO_RE = re.compile(r"from (\(\d+,\s*\d+\)) to (\(\d+,\s*\d+\))")
_MOVES_RE = re.compile(r"After moves ([LRUD]+)", re.IGNORECASE)


def spatial_kpis(tasks, rows):
    """Update ``rows`` with spatial metrics and return aggregates."""
    total = len(rows)
    success = 0
    subopt_sum = 0.0
    steps_sum = 0
    counted = 0
    for task, row in zip(tasks, rows):
        prompt = task.prompt
        pred = row.get("pred", "")
        gmatch = _GRID_RE.search(prompt)
        size = int(gmatch.group(1)) if gmatch else 5
        obstacles: set[tuple[int, int]] = set()
        if gmatch:
            try:
                import ast

                obs_list = ast.literal_eval(gmatch.group(2))
                obstacles = {tuple(map(int, o)) for o in obs_list}
            except Exception:
                obstacles = set()
        start, goal = None, None
        ft = _FROM_TO_RE.search(prompt)
        if ft:
            start = _parse_coord(ft.group(1))
            goal = _parse_coord(ft.group(2))
        else:
            sm = _START_RE.search(prompt)
            gm = _GOAL_RE.search(prompt)
            if sm:
                start = _parse_coord(sm.group(1))
            if gm:
                goal = _parse_coord(gm.group(1))
        moves_prompt = _MOVES_RE.search(prompt)
        if moves_prompt and start is not None:
            path_moves = list(moves_prompt.group(1))
            final = start
            for step in path_moves:
                dx, dy = _MOVE_DIRS[step]
                final = (final[0] + dx, final[1] + dy)
            pred_coord = _parse_coord(pred)
            row["steps_pred"] = len(path_moves)
            row["steps_opt"] = len(path_moves)
            row["suboptimality"] = 1.0
            row["success"] = pred_coord == final
            success += int(row["success"])
            subopt_sum += row["suboptimality"]
            steps_sum += len(path_moves)
            counted += 1
            continue
        if start is not None and goal is not None:
            opt_path = _bfs(start, goal, size, obstacles) or []
            opt_len = len(opt_path)
            pred_moves: List[str] = []
            if "shortest path length" in prompt.lower():
                try:
                    pred_len = int(re.findall(r"-?\d+", pred)[0])
                except Exception:
                    pred_len = 0
                row["steps_pred"] = pred_len
                row["steps_opt"] = opt_len
                row["suboptimality"] = (pred_len / opt_len) if opt_len else 0.0
                row["success"] = pred_len == opt_len
            else:
                pred_moves = _parse_moves(pred)
                row["steps_pred"] = len(pred_moves)
                row["steps_opt"] = opt_len
                row["suboptimality"] = (len(pred_moves) / opt_len) if opt_len else 0.0
                pos = start
                valid = True
                for step in pred_moves:
                    dx, dy = _MOVE_DIRS[step]
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if not (0 <= nx < size and 0 <= ny < size) or (nx, ny) in obstacles:
                        valid = False
                        break
                    pos = (nx, ny)
                row["success"] = valid and pos == goal
            success += int(row["success"])
            subopt_sum += row["suboptimality"]
            steps_sum += row["steps_pred"]
            counted += 1
            continue
        row["success"] = False
        row["steps_pred"] = 0
        row["steps_opt"] = 0
        row["suboptimality"] = 0.0
    metrics = {
        "success_rate": success / total if total else 0.0,
        "suboptimality_ratio": subopt_sum / counted if counted else 0.0,
        "steps_to_goal": steps_sum / counted if counted else 0.0,
    }
    return metrics
