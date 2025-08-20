"""Deterministic place graph with lightweight planning utilities.

Summary
-------
Implements a small topological map used by the spatial/procedural memory
module.  Context strings deterministically become nodes whose edges carry
transition cost and success probability.  Planning uses A* or Dijkstra to
produce optimal paths.
Side Effects
------------
Maintenance helpers may write to an optional log file.

Complexity
----------
Observation is ``O(1)``; planning is ``O(E log V)``.

Examples
--------
>>> g = PlaceGraph()
>>> g.observe("a"); g.observe("b")
1
>>> g.plan("a", "b")
['a', 'b']

See Also
--------
hippo_mem.spatial.algorithm_card
hippo_mem.spatial.macros
"""

from __future__ import annotations

import copy
import heapq
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Edge:
    """Connection metadata between two places.

    Summary
    -------
    Stores edge cost, success probability, and last observation step.

    Parameters
    ----------
    cost : float, optional
        Transition cost; arbitrary units, by default ``1.0``.
    success : float, optional
        Probability of success in ``[0, 1]``, by default ``1.0``.
    last_seen : int, optional
        Step index when edge was last traversed, by default ``0``.
    Examples
    --------
    >>> Edge()
    Edge(cost=1.0, success=1.0, last_seen=0)

    See Also
    --------
    PlaceGraph
    """

    cost: float = 1.0
    success: float = 1.0
    last_seen: int = 0


@dataclass
class Place:
    """A place in the environment with pseudo coordinates.

    Summary
    -------
    Holds coordinate and recency information for a context.

    Parameters
    ----------
    name : str
        Context string.
    coord : Tuple[float, float]
        Pseudo coordinates ``(x, y)``.
    last_seen : int, optional
        Step index of last observation, by default ``0``.
    Examples
    --------
    >>> Place("a", (0.0, 0.0))
    Place(name='a', coord=(0.0, 0.0), last_seen=0)

    See Also
    --------
    Edge
    """

    name: str
    coord: Tuple[float, float]
    last_seen: int = 0


class ContextEncoder:
    """Deterministic context-to-coordinate encoder.

    Summary
    -------
    Maps strings to pseudo coordinates derived from their hash so that
    map growth is deterministic across runs.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Place] = {}

    def encode(self, context: str) -> Place:
        """Return ``Place`` for ``context``.

        Summary
        -------
        Assigns a stable pseudo coordinate to ``context`` and caches the
        result.

        Parameters
        ----------
        context : str
            Textual identifier for the place.

        Returns
        -------
        Place
            Encoded place entry.
        Side Effects
        ------------
        Cache grows with unique contexts.
        Examples
        --------
        >>> enc = ContextEncoder(); enc.encode("a").name
        'a'

        See Also
        --------
        Place
        """

        if context not in self._cache:
            h = hash(context)
            x = (h & 0xFFFF) / 1000.0
            y = ((h >> 16) & 0xFFFF) / 1000.0
            self._cache[context] = Place(context, (x, y))
        return self._cache[context]


class PlaceGraph:
    """Graph of observed places with lightweight planning.

    Summary
    -------
    Maintains nodes for contexts, supports deterministic growth via
    :meth:`observe`, and plans optimal paths using A* or Dijkstra.
    """

    def __init__(self, path_integration: bool = False, *, config: Optional[dict] = None) -> None:
        """Initialise the graph.

        Summary
        -------
        Create an empty map optionally enabling path integration to track
        relative movement.

        Parameters
        ----------
        path_integration : bool, optional
            If ``True``, update coordinates by relative displacements.
        config : dict, optional
            Maintenance settings such as ``decay_rate``.
        Side Effects
        ------------
        May spawn maintenance thread later.
        Examples
        --------
        >>> PlaceGraph()
        <class 'hippo_mem.spatial.map.PlaceGraph'>

        See Also
        --------
        ContextEncoder
        """

        self.encoder = ContextEncoder()
        self._context_to_id: Dict[str, int] = {}
        self._id_to_context: Dict[int, str] = {}
        # adjacency list: node -> neighbour -> Edge
        self.graph: Dict[int, Dict[int, Edge]] = {}
        self._next_id = 0
        self._last_obs: Optional[int] = None
        self._step = 0
        self._path_integration = path_integration
        self._position = (0.0, 0.0)
        self._last_coord: Optional[Tuple[float, float]] = None
        self.config = config or {}
        self._log = {"writes": 0, "recalls": 0, "hits": 0, "maintenance": 0}
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._history: List[dict[str, Any]] = []
        self._max_undo = int(self.config.get("max_undo", 5))
        self._maintenance_log: List[dict[str, Any]] = []
        self._log_file = self.config.get("maintenance_log")

    # ------------------------------------------------------------------
    # Observation and graph construction
    def _ensure_node(self, context: str) -> int:
        """Return node id for ``context`` creating it if absent."""

        node = self._context_to_id.get(context)
        if node is None:
            node = self._next_id
            self._next_id += 1
            self._context_to_id[context] = node
            self._id_to_context[node] = context
            self.graph.setdefault(node, {})
            # why: cache coordinates for deterministic heuristics
            self.encoder.encode(context)
        return node

    # ------------------------------------------------------------------
    # Maintenance helpers
    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "cache": copy.deepcopy(self.encoder._cache),
            "c2id": dict(self._context_to_id),
            "id2c": dict(self._id_to_context),
            "graph": copy.deepcopy(self.graph),
            "next_id": self._next_id,
            "last_obs": self._last_obs,
            "step": self._step,
            "position": self._position,
            "last_coord": self._last_coord,
        }

    def _log_event(self, op: str, info: dict[str, Any]) -> None:
        event = {"ts": time.time(), "op": op, **info}
        self._maintenance_log.append(event)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")

    def _push_history(self, op: str) -> None:
        self._history.append({"op": op, "state": self._snapshot_state()})
        if len(self._history) > self._max_undo:
            self._history.pop(0)

    def observe(self, context: str) -> int:
        """Insert ``context`` into the graph.

        Summary
        -------
        Grows the map deterministically and links consecutive observations
        with undirected unit-cost edges.

        Parameters
        ----------
        context : str
            Context to record.

        Returns
        -------
        int
            Node identifier assigned to ``context``.
        Side Effects
        ------------
        Updates internal step counter and edge timestamps.
        Examples
        --------
        >>> g = PlaceGraph(); g.observe("a")
        0

        See Also
        --------
        connect
        plan
        """

        self._step += 1
        node = self._ensure_node(context)

        place = self.encoder.encode(context)
        if self._path_integration:
            coord = place.coord
            if self._last_coord is None:
                self._position = (0.0, 0.0)
                self._last_coord = coord
            else:
                # why: integrate displacement for path integration
                dx = coord[0] - self._last_coord[0]
                dy = coord[1] - self._last_coord[1]
                self._position = (self._position[0] + dx, self._position[1] + dy)
                self._last_coord = coord
            place.coord = self._position
        place.last_seen = self._step

        if self._last_obs is not None and self._last_obs != node:
            self._add_edge(self._last_obs, node, step=self._step)
            self._add_edge(node, self._last_obs, step=self._step)
        self._last_obs = node
        self._log["writes"] += 1
        return node

    def connect(
        self,
        a_context: str,
        b_context: str,
        cost: float = 1.0,
        success: float = 1.0,
    ) -> None:
        """Explicitly connect two contexts.

        Summary
        -------
        Add bidirectional edge with given ``cost`` and ``success``.

        Parameters
        ----------
        a_context, b_context : str
            Endpoints to connect.
        cost : float, optional
            Transition cost, by default ``1.0``.
        success : float, optional
            Success probability, by default ``1.0``.
        Side Effects
        ------------
        Updates edge timestamps.
        Examples
        --------
        >>> g = PlaceGraph(); g.connect("a", "b")

        See Also
        --------
        observe
        """

        a = self._ensure_node(a_context)
        b = self._ensure_node(b_context)
        self._add_edge(a, b, cost, success)
        self._add_edge(b, a, cost, success)
        self._log["writes"] += 1

    def _add_edge(
        self,
        a: int,
        b: int,
        cost: float = 1.0,
        success: float = 1.0,
        *,
        step: Optional[int] = None,
    ) -> None:
        # why: track last_seen for TTL pruning
        edge = self.graph[a].get(b)
        if edge is None:
            self.graph[a][b] = Edge(cost, success, last_seen=step or self._step)
        else:
            edge.cost = cost
            edge.success = success
            edge.last_seen = step or self._step

    # ------------------------------------------------------------------
    # Planning utilities
    def plan(self, start: str, goal: str, method: str = "astar") -> List[str]:
        """Return the shortest path between contexts.

        Summary
        -------
        Use A* or Dijkstra to compute an optimal path from ``start`` to ``goal``.

        Parameters
        ----------
        start, goal : str
            Start and goal contexts.
        method : str, optional
            ``"astar"`` (default) or ``"dijkstra"``.

        Returns
        -------
        List[str]
            Sequence of contexts including start and goal. Empty list if
            disconnected.

        Raises
        ------
        ValueError
            If ``method`` is unknown.

        Side Effects
        ------------
        Increments recall counters and hit statistics.

        Complexity
        ----------
        ``O(E log V)``.

        Examples
        --------
        >>> g = PlaceGraph(); g.observe("a"); g.observe("b")
        >>> g.plan("a", "b")
        ['a', 'b']

        See Also
        --------
        _a_star
        _dijkstra
        """

        s = self._ensure_node(start)
        g = self._ensure_node(goal)
        if method == "astar":
            path_ids = self._a_star(s, g)
        elif method == "dijkstra":
            path_ids = self._dijkstra(s, g)
        else:
            raise ValueError(f"Unknown planning method: {method}")
        self._log["recalls"] += 1
        if path_ids:
            self._log["hits"] += 1
        return [self._id_to_context[i] for i in path_ids]

    # -- heuristics -----------------------------------------------------
    def _heuristic(self, a: int, b: int) -> float:
        pa = self.encoder.encode(self._id_to_context[a]).coord
        pb = self.encoder.encode(self._id_to_context[b]).coord
        return math.dist(pa, pb)

    def _reconstruct(self, came_from: Dict[int, int], current: int) -> List[int]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _a_star(self, start: int, goal: int) -> List[int]:
        open_set: List[Tuple[float, int]] = [(0.0, start)]
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {start: 0.0}
        f_score: Dict[int, float] = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)
            for neighbor, edge in self.graph.get(current, {}).items():
                tentative = g_score[current] + edge.cost
                if tentative < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def _dijkstra(self, start: int, goal: int) -> List[int]:
        queue: List[Tuple[float, int]] = [(0.0, start)]
        came_from: Dict[int, int] = {}
        costs: Dict[int, float] = {start: 0.0}
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                return self._reconstruct(came_from, node)
            if cost > costs.get(node, math.inf):
                continue
            for neighbor, edge in self.graph.get(node, {}).items():
                new_cost = cost + edge.cost
                if new_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    came_from[neighbor] = node
                    heapq.heappush(queue, (new_cost, neighbor))
        return []

    # ------------------------------------------------------------------
    # Maintenance and logging
    def decay(self, rate: float) -> None:
        """Exponentially decay all positions toward the origin.

        Summary
        -------
        Shrinks coordinates to model positional drift.

        Parameters
        ----------
        rate : float
            Decay rate ``0–1``.
        Side Effects
        ------------
        Records operation for rollback.

        Complexity
        ----------
        ``O(n)`` over stored places.

        Examples
        --------
        >>> g = PlaceGraph(); g.observe("a")
        >>> before = g.encoder.encode("a").coord
        >>> g.decay(0.5)
        >>> g.encoder.encode("a").coord != before
        True

        See Also
        --------
        prune
        rollback
        """

        self._push_history("decay")
        factor = max(0.0, 1.0 - rate)
        # why: dampen coordinates to limit drift
        for place in self.encoder._cache.values():
            x, y = place.coord
            place.coord = (x * factor, y * factor)
        px, py = self._position
        self._position = (px * factor, py * factor)
        if self._last_coord is not None:
            lx, ly = self._last_coord
            self._last_coord = (lx * factor, ly * factor)
        self._log_event("decay", {"rate": rate})

    # ------------------------------------------------------------------
    # Pruning helpers
    def _prune_nodes(self, threshold: int) -> None:
        """Remove nodes whose last observation predates ``threshold``."""

        for context, place in list(self.encoder._cache.items()):
            if place.last_seen < threshold:
                node = self._context_to_id.pop(context, None)
                if node is not None:
                    self._id_to_context.pop(node, None)
                    self.graph.pop(node, None)
                    for nbrs in self.graph.values():
                        nbrs.pop(node, None)
                del self.encoder._cache[context]

    def _prune_edges(self, threshold: int) -> None:
        """Remove edges older than ``threshold`` and drop isolated nodes."""

        for a in list(self.graph.keys()):
            for b in list(self.graph[a].keys()):
                if self.graph[a][b].last_seen < threshold or b not in self._id_to_context:
                    del self.graph[a][b]
            if not self.graph[a]:
                context = self._id_to_context.pop(a, None)
                if context is not None:
                    self._context_to_id.pop(context, None)
                del self.graph[a]

    def prune(self, max_age: int) -> None:
        """Drop edges and places not observed within ``max_age`` steps.

        Summary
        -------
        Prevents unbounded map growth by removing stale items.

        Parameters
        ----------
        max_age : int
            Maximum allowed age in steps.
        Side Effects
        ------------
        Records snapshot for rollback.

        Complexity
        ----------
        ``O(n)`` over nodes and edges.

        Examples
        --------
        >>> g = PlaceGraph(); g.observe("a")
        >>> g.prune(max_age=0)

        See Also
        --------
        decay
        rollback
        """

        self._push_history("prune")
        threshold = self._step - max_age
        # why: drop stale nodes to bound map size
        self._prune_nodes(threshold)
        self._prune_edges(threshold)
        self._log_event("prune", {"max_age": max_age})

    def log_status(self) -> dict:
        """Return counters for writes, recalls, hits, and maintenance."""

        return dict(self._log)

    def start_background_tasks(self, interval: float = 100.0) -> None:
        """Launch periodic decay/prune thread.

        Summary
        -------
        Offloads maintenance to a daemon thread.

        Parameters
        ----------
        interval : float, optional
            Sleep interval in seconds, by default ``100.0``.
        Side Effects
        ------------
        Spawns a background thread.
        Examples
        --------
        >>> g = PlaceGraph(); g.start_background_tasks()  # doctest: +ELLIPSIS

        See Also
        --------
        decay
        prune
        """

        if self._bg_thread is not None:
            return

        stop_event = threading.Event()

        def loop() -> None:
            while not stop_event.wait(interval):
                rate = float(self.config.get("decay_rate", 0.0))
                if rate > 0:
                    self.decay(rate)
                cfg = self.config.get("prune", {})
                age = cfg.get("max_age")
                if age is not None:
                    self.prune(int(age))
                self._log["maintenance"] += 1

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._stop_event = stop_event
        self._bg_thread = t

    def stop_background_tasks(self) -> None:
        """Stop background maintenance thread if running.

        Summary
        -------
        Idempotently signals the maintenance loop to exit and waits
        briefly for the thread to terminate.
        """

        if self._bg_thread is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        self._bg_thread.join(timeout=1.0)
        self._bg_thread = None
        self._stop_event = None

    def rollback(self, n: int = 1) -> None:
        """Rollback the last ``n`` maintenance operations.

        Summary
        -------
        Restore state snapshots captured before decay or prune.

        Parameters
        ----------
        n : int, optional
            Number of operations to undo, by default ``1``.
        Side Effects
        ------------
        Alters internal structures and logs a rollback event.

        Complexity
        ----------
        ``O(n)`` relative to number of stored snapshots.

        Examples
        --------
        >>> g = PlaceGraph(); g.observe("a"); g.decay(0.1)
        >>> g.rollback(1)

        See Also
        --------
        decay
        prune
        """

        for _ in range(n):
            if not self._history:
                break
            entry = self._history.pop()
            state = entry.get("state")
            if state is None:
                continue
            self.encoder._cache = state["cache"]
            self._context_to_id = state["c2id"]
            self._id_to_context = state["id2c"]
            self.graph = state["graph"]
            self._next_id = state["next_id"]
            self._last_obs = state["last_obs"]
            self._step = state["step"]
            self._position = state["position"]
            self._last_coord = state["last_coord"]
            self._log_event("rollback", {"op": entry.get("op")})


__all__ = ["PlaceGraph", "ContextEncoder", "Edge", "Place"]
