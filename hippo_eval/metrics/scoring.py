"""Scoring helpers with exact-match variants and token F1."""

from __future__ import annotations

import heapq
import re
import string
from collections import Counter
from typing import List, Protocol

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_MOVE_SET = {"U", "D", "L", "R"}

_SHORT_ANS_RE = re.compile(r"^[A-Za-z0-9 ,'-]+$")
_UDLR_RE = re.compile(r"^[UDLR]{1,64}$")


def enforce_short_answer(text: str, max_len: int = 64) -> str:
    """Return ``text`` if it satisfies the short-answer policy else ``""``.

    Parameters
    ----------
    text:
        Candidate answer from the model.
    max_len:
        Maximum allowed character length after stripping whitespace.
    """

    text = text.strip()
    if not text or len(text) > max_len or not _SHORT_ANS_RE.fullmatch(text):
        return ""
    return text


def enforce_udlr(text: str, max_len: int = 64) -> str:
    """Return uppercased path if it matches ``UDLR`` policy else ``""``."""

    text = text.strip().upper()
    if _UDLR_RE.fullmatch(text):
        return text
    return ""


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

    Non ``UDLR`` characters are stripped from both strings when the gold
    answer encodes a move sequence. This accepts comma or whitespace
    separated paths so evaluation reflects planning, not formatting.

    ``em_norm`` is otherwise constrained so that a raw mismatch cannot
    yield a normalized match.
    """
    pm = "".join(ch for ch in pred.upper() if ch in _MOVE_SET)
    gm = "".join(ch for ch in gold.upper() if ch in _MOVE_SET)
    allowed = _MOVE_SET | set(string.whitespace) | set(string.punctuation)
    if gm and all(ch in allowed for ch in gold.upper()):
        return int(pm == gm)
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


def _astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    size: int,
    obstacles: set[tuple[int, int]],
) -> list[str] | None:
    """A* search for unit-cost grid worlds."""

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set: list[tuple[int, int, tuple[int, int], list[str]]] = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, []))
    seen: dict[tuple[int, int], int] = {start: 0}
    while open_set:
        _f, g, (x, y), path = heapq.heappop(open_set)
        if (x, y) == goal:
            return path
        for step, (dx, dy) in _MOVE_DIRS.items():
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if 0 <= nx < size and 0 <= ny < size and nxt not in obstacles:
                ng = g + 1
                if ng < seen.get(nxt, float("inf")):
                    seen[nxt] = ng
                    heapq.heappush(open_set, (ng + heuristic(nxt, goal), ng, nxt, path + [step]))
    return None


_GRID_RE = re.compile(r"Grid (\d+)x\1 with obstacles (\[[^]]*\])")
_START_RE = re.compile(r"Start (\(\d+,\s*\d+\))")
_GOAL_RE = re.compile(r"goal (\(\d+,\s*\d+\))")
_FROM_TO_RE = re.compile(r"from (\(\d+,\s*\d+\)) to (\(\d+,\s*\d+\))")
_MOVES_RE = re.compile(r"After moves ([LRUD]+)", re.IGNORECASE)


class SpatialTaskStrategy(Protocol):
    """Strategy interface for spatial task scoring."""

    def evaluate(
        self,
        pred: str,
        start: tuple[int, int],
        goal: tuple[int, int] | None,
        size: int,
        obstacles: set[tuple[int, int]],
        row: dict,
    ) -> None:
        """Populate ``row`` with metrics for a prediction."""


class _MetricAccumulator:
    """Aggregate success, step, and suboptimality metrics."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.success = 0
        self.subopt_sum = 0.0
        self.steps_sum = 0
        self.counted = 0

    def add(self, row: dict) -> None:
        self.success += int(row["success"])
        self.subopt_sum += row["suboptimality"]
        self.steps_sum += row["steps_pred"]
        self.counted += 1

    def metrics(self) -> dict[str, float]:
        return {
            "success_rate": self.success / self.total if self.total else 0.0,
            "suboptimality_ratio": (self.subopt_sum / self.counted if self.counted else 0.0),
            "steps_to_goal": self.steps_sum / self.counted if self.counted else 0.0,
        }


class _MultiMetricAccumulator:
    """Aggregate spatial metrics with optimality gap and plan length."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.success = 0
        self.subopt_sum = 0.0
        self.steps_pred = 0
        self.steps_opt = 0
        self.counted = 0

    def add(self, row: dict) -> None:
        self.success += int(row["success"])
        self.subopt_sum += row["suboptimality"]
        self.steps_pred += row["steps_pred"]
        self.steps_opt += row["steps_opt"]
        self.counted += 1

    def metrics(self) -> dict[str, float]:
        return {
            "success_rate": self.success / self.total if self.total else 0.0,
            "suboptimality_ratio": (self.subopt_sum / self.counted if self.counted else 0.0),
            "mean_plan_length": (self.steps_pred / self.counted if self.counted else 0.0),
            "steps_to_goal": (self.steps_opt / self.counted if self.counted else 0.0),
            "optimality_gap": (
                (self.steps_pred - self.steps_opt) / self.counted if self.counted else 0.0
            ),
        }


def _extract_metadata(
    prompt: str,
) -> tuple[
    int, set[tuple[int, int]], tuple[int, int] | None, tuple[int, int] | None, List[str] | None
]:
    """Return grid size, obstacles, start, goal, and move trace."""

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
    path_moves = list(moves_prompt.group(1)) if moves_prompt else None
    return size, obstacles, start, goal, path_moves


def _default_row(row: dict) -> None:
    row["success"] = False
    row["steps_pred"] = 0
    row["steps_opt"] = 0
    row["suboptimality"] = 0.0
    row["oracle_path"] = None
    row["oracle_success"] = False
    row["pred_matches_oracle"] = False


class _MoveTraceStrategy:
    """Evaluate final coordinate after following a move trace."""

    def __init__(self, path_moves: List[str]) -> None:
        self.path_moves = path_moves

    def evaluate(
        self,
        pred: str,
        start: tuple[int, int],
        goal: tuple[int, int] | None,
        size: int,
        obstacles: set[tuple[int, int]],
        row: dict,
    ) -> None:
        final = start
        for step in self.path_moves:
            dx, dy = _MOVE_DIRS[step]
            final = (final[0] + dx, final[1] + dy)
        pred_coord = _parse_coord(pred)
        row["steps_pred"] = len(self.path_moves)
        row["steps_opt"] = len(self.path_moves)
        row["suboptimality"] = 1.0
        row["success"] = pred_coord == final


class _PathLengthStrategy:
    """Evaluate predicted length of optimal path."""

    def evaluate(
        self,
        pred: str,
        start: tuple[int, int],
        goal: tuple[int, int],
        size: int,
        obstacles: set[tuple[int, int]],
        row: dict,
    ) -> None:
        opt_len = len(_astar(start, goal, size, obstacles) or [])
        try:
            pred_len = int(re.findall(r"-?\d+", pred)[0])
        except Exception:
            pred_len = 0
        row["steps_pred"] = pred_len
        row["steps_opt"] = opt_len
        row["suboptimality"] = (pred_len / opt_len) if opt_len else 0.0
        row["success"] = pred_len == opt_len


class _PathSequenceStrategy:
    """Evaluate a predicted move sequence."""

    def evaluate(
        self,
        pred: str,
        start: tuple[int, int],
        goal: tuple[int, int],
        size: int,
        obstacles: set[tuple[int, int]],
        row: dict,
    ) -> None:
        opt_len = len(_astar(start, goal, size, obstacles) or [])
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
        row["success"] = valid and pos == goal and row["suboptimality"] <= 1.2


def spatial_kpis(tasks, rows):
    """Update ``rows`` with spatial metrics and return aggregates."""

    acc = _MetricAccumulator(len(rows))
    oracle_ok = 0
    valid = 0
    for task, row in zip(tasks, rows):
        prompt = task.prompt
        pred = row.get("pred", "")
        size, obstacles, start, goal, path_moves = _extract_metadata(prompt)
        if path_moves and start is not None:
            strategy: SpatialTaskStrategy = _MoveTraceStrategy(path_moves)
        elif start is None or goal is None:
            _default_row(row)
            continue
        elif "shortest path length" in prompt.lower():
            strategy = _PathLengthStrategy()
        else:
            strategy = _PathSequenceStrategy()
        strategy.evaluate(pred, start, goal, size, obstacles, row)
        oracle = _astar(start, goal, size, obstacles) if start and goal else None
        row["oracle_path"] = "".join(oracle) if oracle else None
        row["oracle_success"] = bool(oracle)
        norm = row.get("normalized_pred") or ""
        row["pred_matches_oracle"] = bool(oracle) and norm == row["oracle_path"]
        oracle_ok += int(bool(oracle))
        valid += int(bool(norm))
        acc.add(row)
    metrics = acc.metrics()
    total = len(rows)
    metrics["valid_action_rate"] = valid / total if total else 0.0
    metrics["oracle_success_rate"] = oracle_ok / total if total else 0.0
    return metrics


def spatial_multi_kpis(tasks, rows):
    """Update ``rows`` with metrics and per-episode curves for spatial_multi."""

    acc = _MultiMetricAccumulator(len(rows))
    per_ep: dict[str, tuple[int, int]] = {}
    oracle_ok = 0
    valid = 0
    for task, row in zip(tasks, rows):
        prompt = task.prompt
        pred = row.get("pred", "")
        size, obstacles, start, goal, path_moves = _extract_metadata(prompt)
        if path_moves and start is not None:
            strategy: SpatialTaskStrategy = _MoveTraceStrategy(path_moves)
        elif start is None or goal is None:
            _default_row(row)
            continue
        elif "shortest path length" in prompt.lower():
            strategy = _PathLengthStrategy()
        else:
            strategy = _PathSequenceStrategy()
        strategy.evaluate(pred, start, goal, size, obstacles, row)
        oracle = _astar(start, goal, size, obstacles) if start and goal else None
        row["oracle_path"] = "".join(oracle) if oracle else None
        row["oracle_success"] = bool(oracle)
        norm = row.get("normalized_pred") or ""
        row["pred_matches_oracle"] = bool(oracle) and norm == row["oracle_path"]
        oracle_ok += int(bool(oracle))
        valid += int(bool(norm))
        acc.add(row)
        ep = getattr(task, "episode_id", None) or "0"
        succ, tot = per_ep.get(ep, (0, 0))
        per_ep[ep] = (succ + int(row["success"]), tot + 1)
    metrics = acc.metrics()
    for idx, ep in enumerate(sorted(per_ep), start=1):
        succ, tot = per_ep[ep]
        metrics[f"success_ep{idx}"] = succ / tot if tot else 0.0
    total = len(rows)
    metrics["valid_action_rate"] = valid / total if total else 0.0
    metrics["oracle_success_rate"] = oracle_ok / total if total else 0.0
    return metrics
