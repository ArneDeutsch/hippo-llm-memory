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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hippo_mem.common.io as io
from hippo_mem.common.history import HistoryEntry, RollbackMixin
from hippo_mem.common.lifecycle import StoreLifecycleMixin


@dataclass
class Edge:
    """Connection metadata between two places.

    Summary
    -------
    Stores edge cost, success probability, weight, last observation step and
    adjacency type.

    Parameters
    ----------
    cost : float, optional
        Transition cost; arbitrary units, by default ``1.0``.
    success : float, optional
        Probability of success in ``[0, 1]``, by default ``1.0``.
    last_seen : int, optional
        Step index when edge was last traversed, by default ``0``.
    weight : float, optional
        Evidence count for the edge, by default ``1.0``.
    kind : str, optional
        Adjacency label such as ``"room"`` or ``"door"``.
    Examples
    --------
    >>> Edge()
    Edge(cost=1.0, success=1.0, last_seen=0, weight=1.0, kind='generic')

    See Also
    --------
    PlaceGraph
    """

    cost: float = 1.0
    success: float = 1.0
    last_seen: int = 0
    weight: float = 1.0
    kind: str = "generic"


@dataclass
class Place:
    """A place in the environment with pseudo coordinates.

    Summary
    -------
    Holds coordinate, type and recency information for a context.

    Parameters
    ----------
    name : str
        Context string.
    coord : Tuple[float, float]
        Pseudo coordinates ``(x, y)`` normalised to ``[0, 1]``.
    last_seen : int, optional
        Step index of last observation, by default ``0``.
    kind : str, optional
        Landmark type such as ``"room"`` or ``"object"``.
    Examples
    --------
    >>> Place("a", (0.0, 0.0))
    Place(name='a', coord=(0.0, 0.0), last_seen=0, kind='generic')

    See Also
    --------
    Edge
    """

    name: str
    coord: Tuple[float, float]
    last_seen: int = 0
    kind: str = "generic"


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
            x = (h & 0xFFFF) / 65535.0
            y = ((h >> 16) & 0xFFFF) / 65535.0
            self._cache[context] = Place(context, (x, y))
        return self._cache[context]


class PlaceGraph(StoreLifecycleMixin, RollbackMixin):
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
        self._log = {
            "writes": 0,
            "recalls": 0,
            "hits": 0,
            "maintenance": 0,
            "landmarks_added": 0,
            "edges_added": 0,
        }
        self._maintenance_log: List[dict[str, Any]] = []
        self._log_file = self.config.get("maintenance_log")
        StoreLifecycleMixin.__init__(self)
        RollbackMixin.__init__(self, int(self.config.get("max_undo", 5)))

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
            self._log["landmarks_added"] += 1
        return node

    def add_landmark(self, context: str, coord: Tuple[float, float], kind: str = "generic") -> int:
        """Insert a landmark with explicit coordinates.

        Parameters
        ----------
        context:
            Landmark identifier.
        coord:
            ``(x, y)`` pair normalised to ``[0, 1]``. Values outside this range
            are clipped.
        kind:
            Landmark type such as ``"room"`` or ``"object"``.
        """

        self._step += 1
        x = max(0.0, min(1.0, float(coord[0])))
        y = max(0.0, min(1.0, float(coord[1])))
        node = self._ensure_node(context)
        place = Place(context, (x, y), self._step, kind)
        self.encoder._cache[context] = place
        self._last_obs = node
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

    def aggregate_duplicate(self, prev_ctx: str, context: str) -> None:
        """Bump edge weight and recency without creating a new edge."""

        self._step += 1
        a = self._ensure_node(prev_ctx)
        b = self._ensure_node(context)
        edge = self.graph[a].get(b)
        if edge is None:
            return
        edge.weight = getattr(edge, "weight", 1.0) + 1.0
        edge.last_seen = self._step
        rev = self.graph[b].get(a)
        if rev is not None:
            rev.weight = getattr(rev, "weight", 1.0) + 1.0
            rev.last_seen = self._step
        self._last_obs = b

    def connect(
        self,
        a_context: str,
        b_context: str,
        cost: float = 1.0,
        success: float = 1.0,
        *,
        kind: str = "generic",
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
        self._add_edge(a, b, cost, success, kind=kind)
        self._add_edge(b, a, cost, success, kind=kind)
        self._log["writes"] += 1

    def _add_edge(
        self,
        a: int,
        b: int,
        cost: float = 1.0,
        success: float = 1.0,
        *,
        step: Optional[int] = None,
        kind: str = "generic",
    ) -> None:
        # why: track last_seen for TTL pruning
        edge = self.graph[a].get(b)
        if edge is None:
            self.graph[a][b] = Edge(
                cost, success, last_seen=step or self._step, weight=1.0, kind=kind
            )
            self._log["edges_added"] += 1
        else:
            edge.cost = cost
            edge.success = success
            edge.last_seen = step or self._step
            edge.kind = kind

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

        self._push_history("decay", self._snapshot_state())
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

        self._push_history("prune", self._snapshot_state())
        threshold = self._step - max_age
        # why: drop stale nodes to bound map size
        self._prune_nodes(threshold)
        self._prune_edges(threshold)
        self._log_event("prune", {"max_age": max_age})

    # ------------------------------------------------------------------
    # Persistence
    def save(
        self,
        directory: str,
        session_id: str,
        fmt: str = "jsonl",
        replay_samples: int = 0,
        gate_attempts: int = 0,
    ) -> None:
        """Save map under ``directory/session_id``."""

        path = Path(directory) / session_id
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "schema": "spatial.store_meta.v1",
            "replay_samples": int(replay_samples),
            "source": (
                "replay" if replay_samples > 0 else "teach" if gate_attempts > 0 else "stub"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        io.atomic_write_json(path / "store_meta.json", meta)
        file = path / "spatial.jsonl"
        if replay_samples <= 0:
            io.atomic_write_file(file, lambda tmp: open(tmp, "w", encoding="utf-8").write(""))
            return

        if fmt == "jsonl":

            def _write(tmp_path: Path) -> None:
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    meta_rec = {
                        "schema": "spatial.v1",
                        "type": "meta",
                        "next_id": self._next_id,
                        "last_obs": self._last_obs,
                        "step": self._step,
                        "position": self._position,
                        "last_coord": self._last_coord,
                    }
                    fh.write(json.dumps(meta_rec) + "\n")
                    for context, place in self.encoder._cache.items():
                        fh.write(
                            json.dumps(
                                {
                                    "schema": "spatial.v1",
                                    "type": "node",
                                    "id": self._context_to_id[context],
                                    "context": context,
                                    "coord": list(place.coord),
                                    "last_seen": place.last_seen,
                                    "kind": place.kind,
                                }
                            )
                            + "\n"
                        )
                    for src, nbrs in self.graph.items():
                        for dst, edge in nbrs.items():
                            fh.write(
                                json.dumps(
                                    {
                                        "schema": "spatial.v1",
                                        "type": "edge",
                                        "src": src,
                                        "dst": dst,
                                        "cost": edge.cost,
                                        "success": edge.success,
                                        "last_seen": edge.last_seen,
                                        "weight": edge.weight,
                                        "kind": edge.kind,
                                    }
                                )
                                + "\n"
                            )

            io.atomic_write_file(file, _write)
        elif fmt == "parquet":
            try:
                import pandas as pd
            except Exception as exc:  # pragma: no cover - optional
                raise RuntimeError("Parquet support requires pandas") from exc
            nodes = []
            for context, place in self.encoder._cache.items():
                nodes.append(
                    {
                        "id": self._context_to_id[context],
                        "context": context,
                        "coord": list(place.coord),
                        "last_seen": place.last_seen,
                        "kind": place.kind,
                    }
                )
            edges = []
            for src, nbrs in self.graph.items():
                for dst, edge in nbrs.items():
                    edges.append(
                        {
                            "src": src,
                            "dst": dst,
                            "cost": edge.cost,
                            "success": edge.success,
                            "last_seen": edge.last_seen,
                            "weight": edge.weight,
                            "kind": edge.kind,
                        }
                    )
            io.atomic_write_file(
                path / "spatial_nodes.parquet",
                lambda tmp: pd.DataFrame(nodes).to_parquet(tmp, index=False),
            )
            io.atomic_write_file(
                path / "spatial_edges.parquet",
                lambda tmp: pd.DataFrame(edges).to_parquet(tmp, index=False),
            )
            meta = {
                "next_id": self._next_id,
                "last_obs": self._last_obs,
                "step": self._step,
                "position": self._position,
                "last_coord": self._last_coord,
            }
            io.atomic_write_json(path / "spatial_meta.json", meta)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {fmt}")

    def load(self, directory: str, session_id: str, fmt: str = "jsonl") -> None:
        """Load map from ``directory/session_id``."""

        path = Path(directory) / session_id
        self.graph.clear()
        self.encoder._cache.clear()
        self._context_to_id.clear()
        self._id_to_context.clear()
        if fmt == "jsonl":
            file = path / "spatial.jsonl"
            for rec in io.read_jsonl(file):
                typ = rec.get("type")
                if typ == "meta":
                    self._next_id = rec.get("next_id", 0)
                    self._last_obs = rec.get("last_obs", 0)
                    self._step = rec.get("step", 0)
                    self._position = tuple(rec.get("position", (0.0, 0.0)))
                    lc = rec.get("last_coord")
                    self._last_coord = tuple(lc) if lc is not None else None
                elif typ == "node":
                    pid = int(rec["id"])
                    context = rec["context"]
                    place = Place(
                        context,
                        tuple(rec["coord"]),
                        int(rec.get("last_seen", 0)),
                        rec.get("kind", "generic"),
                    )
                    self.encoder._cache[context] = place
                    self._context_to_id[context] = pid
                    self._id_to_context[pid] = context
                    self.graph[pid] = {}
                elif typ == "edge":
                    edge = Edge(
                        cost=float(rec["cost"]),
                        success=float(rec["success"]),
                        last_seen=int(rec.get("last_seen", 0)),
                        weight=float(rec.get("weight", 1.0)),
                        kind=rec.get("kind", "generic"),
                    )
                    self.graph.setdefault(int(rec["src"]), {})[int(rec["dst"])] = edge
        elif fmt == "parquet":
            nodes_df = io.read_parquet(path / "spatial_nodes.parquet")
            edges_df = io.read_parquet(path / "spatial_edges.parquet")
            meta = io.read_json(path / "spatial_meta.json")
            for rec in nodes_df.to_dict(orient="records"):
                pid = int(rec["id"])
                context = rec["context"]
                place = Place(
                    context,
                    tuple(rec["coord"]),
                    int(rec.get("last_seen", 0)),
                    rec.get("kind", "generic"),
                )
                self.encoder._cache[context] = place
                self._context_to_id[context] = pid
                self._id_to_context[pid] = context
                self.graph[pid] = {}
            for rec in edges_df.to_dict(orient="records"):
                edge = Edge(
                    cost=float(rec["cost"]),
                    success=float(rec["success"]),
                    last_seen=int(rec.get("last_seen", 0)),
                    weight=float(rec.get("weight", 1.0)),
                    kind=rec.get("kind", "generic"),
                )
                self.graph.setdefault(int(rec["src"]), {})[int(rec["dst"])] = edge
            self._next_id = meta.get("next_id", 0)
            self._last_obs = meta.get("last_obs", 0)
            self._step = meta.get("step", 0)
            self._position = tuple(meta.get("position", (0.0, 0.0)))
            lc = meta.get("last_coord")
            self._last_coord = tuple(lc) if lc is not None else None
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {fmt}")

    def log_status(self) -> dict:
        """Return counters for writes, recalls, hits, and maintenance."""

        return dict(self._log)

    def _maintenance_tick(self, _event: threading.Event) -> None:
        rate = float(self.config.get("decay_rate", 0.0))
        if rate > 0:
            self.decay(rate)
        cfg = self.config.get("prune", {})
        age = cfg.get("max_age")
        if age is not None:
            self.prune(int(age))
        self._log["maintenance"] += 1

    def _apply_rollback(self, entry: HistoryEntry) -> None:
        state = entry.data
        if state is None:
            return
        self.encoder._cache = state["cache"]
        self._context_to_id = state["c2id"]
        self._id_to_context = state["id2c"]
        self.graph = state["graph"]
        self._next_id = state["next_id"]
        self._last_obs = state["last_obs"]
        self._step = state["step"]
        self._position = state["position"]
        self._last_coord = state["last_coord"]
        self._log_event("rollback", {"op": entry.op})


__all__ = ["PlaceGraph", "ContextEncoder", "Edge", "Place"]
