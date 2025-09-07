"""Semantic graph persistence and retrieval.

Summary
-------
Implements a ``KnowledgeGraph`` storing tuples in a NetworkX multigraph
with SQLite persistence. Tuples are routed via a :class:`SchemaIndex`
and enriched with embeddings. Subgraph retrieval expands around
top-matching nodes within a configurable radius.
Side Effects
------------
Creates SQLite files and spawns optional background maintenance
threads.

Complexity
----------
Operations are linear in the number of affected nodes/edges.

Examples
--------
>>> kg = KnowledgeGraph()
>>> kg.upsert('Alice', 'knows', 'Bob', 'Alice knows Bob')
>>> kg.retrieve([1.0, 0.0], k=1).number_of_nodes()
2

See Also
--------
hippo_mem.relational.schema.SchemaIndex
hippo_mem.relational.tuples.extract_tuples
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import networkx as nx
import numpy as np

import hippo_mem.common.io as io
from hippo_mem.common import GateDecision
from hippo_mem.common.history import HistoryEntry, RollbackMixin
from hippo_mem.common.lifecycle import StoreLifecycleMixin
from hippo_mem.retrieval.embed import embed_text

from .backend import PersistenceStrategy, SQLiteBackend
from .gating import RelationalGate
from .maintenance import MaintenanceManager
from .schema import SchemaIndex
from .tuples import TupleType

_log = logging.getLogger(__name__)


def _ensure_vec(text: str, dim: int = 16) -> list[float]:
    """Return a non-zero embedding for ``text``.

    Parameters
    ----------
    text : str
        Text to embed.
    dim : int, optional
        Embedding size, default ``16``.
    """

    vec = embed_text(text, dim=dim)
    if len(vec) != dim or not any(abs(v) > 1e-12 for v in vec):
        raise ValueError("embedding generation failed")
    return vec


class KnowledgeGraph(StoreLifecycleMixin, RollbackMixin):
    """Persistent semantic graph with provenance tracking.

    Summary
    -------
    Stores tuples in a NetworkX ``MultiDiGraph`` and mirrors them in an
    SQLite database for durability. Schema-fit tuples can be fast-tracked
    via :class:`SchemaIndex`.

    Parameters
    ----------
    db_path : str, optional
        SQLite path or ``":memory:"``.
    config : Optional[dict], optional
        Configuration flags such as ``schema_threshold`` and
        ``gnn_updates``.
    Side Effects
    ------------
    Opens a SQLite connection and optionally spawns a background thread.

    See Also
    --------
    SchemaIndex
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        config: Optional[dict] = None,
        gate: Optional[RelationalGate] = None,
        backend: Optional[PersistenceStrategy] = None,
        maintenance: Optional[MaintenanceManager] = None,
    ) -> None:
        self.graph = nx.MultiDiGraph()
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.backend = backend or SQLiteBackend(db_path)
        self.backend.init_db()
        self.backend.load(self.graph, self.node_embeddings)
        self.config = config or {}
        thresh = float(self.config.get("schema_threshold", 0.8))
        self.schema_index = SchemaIndex(threshold=thresh)
        self._gnn_updates = bool(self.config.get("gnn_updates", True))
        self._log = {"writes": 0, "recalls": 0, "hits": 0, "maintenance": 0}
        self._maintenance_log: list[dict[str, Any]] = []
        self._log_file = self.config.get("maintenance_log")
        self.gate = gate
        self._episodic_queue: list[TupleType] = []
        self._maintenance = maintenance or MaintenanceManager(self)
        self.counters = {"nodes_added": 0, "edges_added": 0, "coref_merges": 0}
        self._coref_index: Dict[str, str] = {}
        StoreLifecycleMixin.__init__(self)
        RollbackMixin.__init__(self, int(self.config.get("max_undo", 5)))
        self._backfill_embeddings()

    # ------------------------------------------------------------------
    # Graph manipulation
    def _resolve_name(self, name: str) -> str:
        """Return canonical form for ``name`` and count merges."""

        key = name.lower()
        canon = self._coref_index.get(key)
        if canon is not None:
            if canon != name:
                self.counters["coref_merges"] += 1
            return canon
        self._coref_index[key] = name
        return name

    def upsert(
        self,
        head: str,
        relation: str,
        tail: str,
        context: str,
        head_type: str = "entity",
        tail_type: str = "entity",
        time: Optional[str] = None,
        conf: float = 1.0,
        provenance: Optional[int] = None,
        *,
        head_embedding: Optional[Iterable[float]] = None,
        tail_embedding: Optional[Iterable[float]] = None,
        edge_embedding: Optional[Iterable[float]] = None,
    ) -> None:
        """Add or update a tuple in the graph and SQLite store.

        Summary
        -------
        Inserts nodes/edges, persists them, and optionally updates node
        embeddings via a tiny GNN-like averaging step.

        Parameters
        ----------
        head, relation, tail, context, time, conf, provenance
            Components of the tuple. ``provenance`` is an identifier for
            audit/rollback.
        head_embedding, tail_embedding, edge_embedding : Optional[Iterable[float]]
            Embeddings with shape ``(D,)``.
        Side Effects
        ------------
        Writes to SQLite and mutates ``self.graph``.
        Examples
        --------
        >>> kg = KnowledgeGraph()
        >>> kg.upsert('A', 'rel', 'B', 'ctx')
        >>> kg.graph.has_edge('A', 'B')
        True

        See Also
        --------
        ingest
        """

        head = self._resolve_name(head)
        tail = self._resolve_name(tail)
        if head_embedding is not None:
            head_embedding = list(head_embedding)
            dim = len(head_embedding)
        else:
            dim = self.dim or 16
            head_embedding = _ensure_vec(head, dim)
        if tail_embedding is not None:
            tail_embedding = list(tail_embedding)
            if len(tail_embedding) != dim:
                raise ValueError("tail embedding dimension mismatch")
        else:
            tail_embedding = _ensure_vec(tail, dim)
        if edge_embedding is not None:
            edge_embedding = list(edge_embedding)
            if len(edge_embedding) != dim:
                raise ValueError("edge embedding dimension mismatch")
        else:
            edge_embedding = _ensure_vec(f"{head}:{relation}:{tail}", dim)
        if head not in self.graph:
            self.graph.add_node(head, type=head_type)
            self.counters["nodes_added"] += 1
        if tail not in self.graph:
            self.graph.add_node(tail, type=tail_type)
            self.counters["nodes_added"] += 1

        # why: persist tuple and provenance for rollback
        self.backend.exec(
            "INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)",
            (head, self._to_json(head_embedding)),
        )
        self.backend.exec(
            "INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)",
            (tail, self._to_json(tail_embedding)),
        )
        edge_id = self.backend.exec(
            "INSERT INTO edges(src, relation, dst, context, time, conf, provenance, embedding) VALUES (?,?,?,?,?,?,?,?)",
            (
                head,
                relation,
                tail,
                context,
                time,
                conf,
                provenance,
                self._to_json(edge_embedding),
            ),
            fetch="lastrowid",
        )

        self.graph.add_edge(
            head,
            tail,
            key=edge_id,
            relation=relation,
            context=context,
            time=time,
            conf=conf,
            provenance=provenance,
        )
        self.counters["edges_added"] += 1
        if edge_embedding is not None:
            self.graph[head][tail][edge_id]["embedding"] = np.asarray(
                list(edge_embedding), dtype=float
            )

        if head_embedding is not None:
            self.node_embeddings[head] = np.asarray(list(head_embedding), dtype=float)
        if tail_embedding is not None:
            self.node_embeddings[tail] = np.asarray(list(tail_embedding), dtype=float)
        if self._gnn_updates:
            self._gnn_update([head, tail])
        self._log["writes"] += 1

    def ingest(self, tup: TupleType) -> GateDecision:
        """Route ``tup`` and return the chosen :class:`GateDecision`.

        Summary
        -------
        Delegates to :meth:`SchemaIndex.fast_track` for inserts, aggregates
        duplicates or routes low-score tuples to the episodic queue.

        Parameters
        ----------
        tup : TupleType
            Tuple from :func:`hippo_mem.relational.tuples.extract_tuples`.

        Returns
        -------
        GateDecision
            Action and gate reason.
        Side Effects
        ------------
        May mutate the graph or enqueue tuples for episodic storage.

        Complexity
        ----------
        ``O(#schemas)`` comparison.

        Examples
        --------
        >>> kg = KnowledgeGraph()
        >>> kg.schema_index.add_schema('likes', 'likes')
        >>> kg.ingest(('A', 'likes', 'B', 'ctx', None, 0.9, 0)).action
        'insert'

        See Also
        --------
        SchemaIndex.fast_track
        """
        padded = tup
        if len(padded) == 7:
            head, rel, tail, ctx, time_str, conf, prov = padded
            padded = (head, rel, tail, ctx, time_str, conf, prov, "entity", "entity")

        decision = GateDecision("insert", "no_gate")
        if self.gate:
            decision = self.gate.decide(padded, self)

        if decision.action == "insert":
            self.schema_index.fast_track(tup, self)
        elif decision.action == "aggregate":
            self.aggregate_duplicate(padded)
        elif decision.action == "route_to_episodic":
            self.route_to_episodic(tup)
        return decision

    def _backfill_embeddings(self) -> None:
        """Compute and persist embeddings when missing."""

        dim = self.dim or 16
        missing_nodes = [n for n in self.graph.nodes if n not in self.node_embeddings]
        for n in missing_nodes:
            vec = _ensure_vec(n, dim)
            self.node_embeddings[n] = np.asarray(vec, dtype=float)
            self.backend.exec(
                "INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)",
                (n, json.dumps(vec)),
            )
        if missing_nodes:
            _log.warning("backfilled %d node embeddings", len(missing_nodes))
        missing_edges = 0
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if data.get("embedding") is None:
                vec = _ensure_vec(f"{u}:{data.get('relation')}:{v}", dim)
                data["embedding"] = np.asarray(vec, dtype=float)
                self.backend.exec(
                    "UPDATE edges SET embedding=? WHERE id=?",
                    (json.dumps(vec), k),
                )
                missing_edges += 1
        if missing_edges:
            _log.warning("backfilled %d edge embeddings", missing_edges)

    def aggregate_duplicate(self, tup: TupleType) -> None:
        """Increase edge evidence for an existing tuple."""

        head, rel, tail, ctx, time_str, conf, prov, *_ = tup
        data = self.graph.get_edge_data(head, tail) or {}
        for edge_id, edge in data.items():
            if edge.get("relation") == rel:
                edge["conf"] = edge.get("conf", 0.0) + conf
                edge["time"] = time_str
                self.backend.exec(
                    "UPDATE edges SET conf=?, time=? WHERE id=?",
                    (edge["conf"], time_str, edge_id),
                )
                break

    def route_to_episodic(self, tup: TupleType) -> None:
        """Enqueue ``tup`` for episodic storage."""

        # TODO unify with episodic async writer
        self._episodic_queue.append(tup)

    def _to_json(self, emb: Optional[Iterable[float]]) -> Optional[str]:
        if emb is None:
            return None
        return json.dumps(list(map(float, emb)))

    def _gnn_update(self, nodes: Sequence[str]) -> None:
        """Very small message-passing stub updating node embeddings.

        Summary
        -------
        Averages incident edge embeddings into node embeddings when
        available.

        Parameters
        ----------
        nodes : Sequence[str]
            Node names to update.
        Side Effects
        ------------
        Mutates ``node_embeddings``.

        Complexity
        ----------
        ``O(degree(n))`` per node.

        Examples
        --------
        >>> kg = KnowledgeGraph()
        >>> kg.upsert('A', 'rel', 'B', 'ctx', edge_embedding=[1.0])
        >>> kg._gnn_update(['A'])

        See Also
        --------
        None
        """

        for n in nodes:
            edges = [
                d.get("embedding")
                for _, _, d in self.graph.edges(n, data=True)
                if d.get("embedding") is not None
            ]
            if not edges:
                continue
            mean = np.mean(np.asarray(edges), axis=0)
            if n in self.node_embeddings:
                if self.node_embeddings[n].shape == mean.shape:
                    self.node_embeddings[n] = (self.node_embeddings[n] + mean) / 2.0
                else:
                    self.node_embeddings[n] = mean
            else:
                self.node_embeddings[n] = mean

    # ------------------------------------------------------------------
    # Retrieval
    def retrieve(
        self, query_embedding: Iterable[float], k: int = 1, radius: int = 1
    ) -> nx.MultiDiGraph:
        """Return a radius-``r`` subgraph around the top-``k`` nodes.

        Summary
        -------
        Scores stored node embeddings against ``query_embedding`` and
        expands neighborhoods within ``radius``.

        Parameters
        ----------
        query_embedding : Iterable[float]
            Query vector of shape ``(D,)``.
        k : int, optional
            Number of seed nodes.
        radius : int, optional
            Breadth of subgraph expansion.

        Returns
        -------
        nx.MultiDiGraph
            Detached subgraph.
        Complexity
        ----------
        ``O(|V|)`` for scoring plus expansion cost.

        Examples
        --------
        >>> kg = KnowledgeGraph()
        >>> kg.upsert('A', 'rel', 'B', 'ctx', head_embedding=[1, 0])
        >>> sub = kg.retrieve([1, 0])
        >>> set(sub.nodes()) == {'A', 'B'}
        True

        See Also
        --------
        upsert
        """

        if not self.node_embeddings:
            return nx.MultiDiGraph()
        if len(self.node_embeddings) != self.graph.number_of_nodes():
            raise ValueError("graph contains nodes without embeddings")

        q = np.asarray(list(query_embedding), dtype=float)
        dim = self.dim
        if q.shape[0] != dim:
            if q.shape[0] > dim:
                q = q[:dim]
            else:
                q = np.pad(q, (0, dim - q.shape[0]))
        scores = {n: float(np.dot(vec, q)) for n, vec in self.node_embeddings.items()}
        top = sorted(scores, key=scores.get, reverse=True)[:k]

        nodes = set()
        for n in top:
            if n in self.graph:
                nodes.update(nx.ego_graph(self.graph, n, radius).nodes())

        sub = self.graph.subgraph(nodes).copy()
        self._log["recalls"] += 1
        self._log["hits"] += sub.number_of_nodes()
        return sub

    @property
    def dim(self) -> int:
        """Return dimensionality of stored node embeddings.

        Returns
        -------
        int
            Embedding dimension or ``0`` when no embeddings are present.
        """

        if self.node_embeddings:
            first = next(iter(self.node_embeddings.values()))
            return int(first.shape[0])
        return 0

    # ------------------------------------------------------------------
    # Maintenance helpers
    def _log_event(self, op: str, info: dict[str, Any]) -> None:
        event = {"ts": time.time(), "op": op, **info}
        self._maintenance_log.append(event)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")

    # ------------------------------------------------------------------
    # Maintenance and logging

    def _build_prune_conditions(
        self, min_conf: Optional[float], max_age: Optional[float]
    ) -> tuple[str, list[float]]:
        """Return SQL ``WHERE`` clause and parameters for ``prune``."""

        conditions: list[str] = []
        params: list[float] = []
        if min_conf is not None:
            conditions.append("conf < ?")
            params.append(min_conf)
        if max_age is not None:
            cutoff = time.time() - max_age
            conditions.append("(time IS NOT NULL AND CAST(time AS REAL) < ?)")
            params.append(cutoff)
        return " OR ".join(conditions), params

    def _remove_edges(self, edges: Sequence[tuple]) -> None:
        """Delete ``edges`` from graph and SQLite store."""

        for edge_id, src, _rel, dst, *_ in edges:
            if self.graph.has_edge(src, dst, key=edge_id):
                self.graph.remove_edge(src, dst, key=edge_id)
            self.backend.exec("DELETE FROM edges WHERE id=?", (edge_id,))

    def _remove_orphan_nodes(
        self,
        candidate_nodes: set[str],
        node_data: dict[str, Optional[np.ndarray]],
    ) -> list[tuple[str, Optional[np.ndarray]]]:
        """Delete nodes with no remaining edges and return their data."""

        removed: list[tuple[str, Optional[np.ndarray]]] = []
        for n in list(candidate_nodes):
            if self.graph.degree(n) == 0:
                self.graph.remove_node(n)
                self.backend.exec("DELETE FROM nodes WHERE name=?", (n,))
                removed.append((n, node_data.get(n)))
                self.node_embeddings.pop(n, None)
        return removed

    def prune(self, min_conf: float = 0.0, max_age: Optional[float] = None) -> None:
        """Remove low-confidence or stale edges.

        Summary
        -------
        Deletes edges that do not meet confidence or age criteria and
        cleans up orphan nodes.

        Parameters
        ----------
        min_conf : float, optional
            Minimum confidence to retain.
        max_age : Optional[float], optional
            Maximum age in seconds.
        Side Effects
        ------------
        Mutates graph and writes to SQLite; logs operations for rollback.

        Complexity
        ----------
        ``O(#edges)`` affected.

        Examples
        --------
        >>> kg = KnowledgeGraph()
        >>> kg.upsert('A', 'rel', 'B', 'ctx', conf=0.4)
        >>> kg.prune(min_conf=0.5)
        >>> kg.graph.number_of_edges()
        0

        See Also
        --------
        rollback
        """

        where, params = self._build_prune_conditions(min_conf, max_age)
        if not where:
            return
        edges = self.backend.exec(
            f"SELECT id, src, relation, dst, context, time, conf, provenance, embedding FROM edges WHERE {where}",
            params,
            fetch="all",
        )
        if not edges:
            return
        candidate_nodes = {e[1] for e in edges} | {e[3] for e in edges}
        node_data = {n: self.node_embeddings.get(n) for n in candidate_nodes}
        self._remove_edges(edges)
        removed_nodes = self._remove_orphan_nodes(candidate_nodes, node_data)

        ops = [{"type": "edge", "data": e} for e in edges]
        ops.extend({"type": "node", "data": n} for n in removed_nodes)
        self._push_history("prune", ops)
        # why: track deletions for provenance and undo
        self._log_event("prune", {"min_conf": min_conf, "max_age": max_age})

    def _maintenance_tick(self, event: threading.Event) -> None:
        self._maintenance.tick(event)

    def _apply_rollback(self, entry: HistoryEntry) -> None:
        if entry.op != "prune":
            return
        for op in entry.data:
            if op["type"] == "node":
                self._restore_node(op["data"])
            elif op["type"] == "edge":
                self._restore_edge(op["data"])
        self._log_event("rollback", {"op": "prune"})

    def _restore_node(self, data: tuple) -> None:
        """Recreate a single node from ``rollback`` data."""

        self._seen_nodes: set[str] = getattr(self, "_seen_nodes", set())
        name, emb, *rest = data
        node_type = rest[0] if rest else None
        if name in self._seen_nodes:
            return
        self._seen_nodes.add(name)
        self.graph.add_node(name, type=node_type)
        self.backend.exec(
            "INSERT OR IGNORE INTO nodes(name, embedding) VALUES (?, ?)",
            (name, self._to_json(emb)),
        )
        if emb is not None:
            self.node_embeddings[name] = np.asarray(emb, dtype=float)

    def _restore_edge(self, edge: tuple) -> None:
        """Recreate a single edge from ``rollback`` data."""

        self._seen_edges: set[int] = getattr(self, "_seen_edges", set())
        edge_id, src, rel, dst, ctx, t_, conf, prov, emb = edge
        if edge_id in self._seen_edges:
            return
        self._seen_edges.add(edge_id)
        self.graph.add_edge(
            src,
            dst,
            key=edge_id,
            relation=rel,
            context=ctx,
            time=t_,
            conf=conf,
            provenance=prov,
        )
        if emb is not None:
            self.graph[src][dst][edge_id]["embedding"] = np.asarray(json.loads(emb), dtype=float)
        self.backend.exec(
            "INSERT OR IGNORE INTO edges(id, src, relation, dst, context, time, conf, provenance, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
            (edge_id, src, rel, dst, ctx, t_, conf, prov, emb),
        )

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
        """Save graph to ``directory/session_id``."""

        path = Path(directory) / session_id
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "schema": "relational.store_meta.v1",
            "replay_samples": int(replay_samples),
            "source": (
                "replay" if replay_samples > 0 else "teach" if gate_attempts > 0 else "stub"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        io.atomic_write_json(path / "store_meta.json", meta)
        file = path / "kg.jsonl"
        if fmt == "jsonl":

            def _write(tmp_path: Path) -> None:
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    for name in self.graph.nodes:
                        emb = self.node_embeddings.get(name)
                        node_type = self.graph.nodes[name].get("type")
                        fh.write(
                            json.dumps(
                                {
                                    "schema": "relational.v1",
                                    "type": "node",
                                    "name": name,
                                    "node_type": node_type,
                                    "embedding": emb.tolist() if emb is not None else None,
                                }
                            )
                            + "\n"
                        )
                    for src, dst, key, data in self.graph.edges(data=True, keys=True):
                        emb = data.get("embedding")
                        fh.write(
                            json.dumps(
                                {
                                    "schema": "relational.v1",
                                    "type": "edge",
                                    "id": key,
                                    "src": src,
                                    "relation": data.get("relation"),
                                    "dst": dst,
                                    "context": data.get("context"),
                                    "time": data.get("time"),
                                    "conf": data.get("conf"),
                                    "provenance": data.get("provenance"),
                                    "embedding": emb.tolist() if emb is not None else None,
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
            for name in self.graph.nodes:
                emb = self.node_embeddings.get(name)
                nodes.append(
                    {
                        "name": name,
                        "node_type": self.graph.nodes[name].get("type"),
                        "embedding": emb.tolist() if emb is not None else None,
                    }
                )
            edges = []
            for src, dst, key, data in self.graph.edges(data=True, keys=True):
                emb = data.get("embedding")
                edges.append(
                    {
                        "id": key,
                        "src": src,
                        "relation": data.get("relation"),
                        "dst": dst,
                        "context": data.get("context"),
                        "time": data.get("time"),
                        "conf": data.get("conf"),
                        "provenance": data.get("provenance"),
                        "embedding": emb.tolist() if emb is not None else None,
                    }
                )
            io.atomic_write_file(
                path / "relational_nodes.parquet",
                lambda tmp: pd.DataFrame(nodes).to_parquet(tmp, index=False),
            )
            io.atomic_write_file(
                path / "relational_edges.parquet",
                lambda tmp: pd.DataFrame(edges).to_parquet(tmp, index=False),
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {fmt}")

    def load(self, directory: str, session_id: str, fmt: str = "jsonl") -> None:
        """Load graph from ``directory/session_id``."""

        path = Path(directory) / session_id
        if fmt == "jsonl":
            file = path / "kg.jsonl"
            if not file.exists():
                file = path / "relational.jsonl"
            for rec in io.read_jsonl(file):
                if rec.get("type") == "node":
                    emb = (
                        np.asarray(rec.get("embedding"), dtype=float)
                        if rec.get("embedding") is not None
                        else None
                    )
                    node_type = rec.get("node_type")
                    self._restore_node((rec["name"], emb, node_type))
                elif rec.get("type") == "edge":
                    emb_json = (
                        json.dumps(rec.get("embedding"))
                        if rec.get("embedding") is not None
                        else None
                    )
                    self._restore_edge(
                        (
                            rec["id"],
                            rec["src"],
                            rec["relation"],
                            rec["dst"],
                            rec.get("context"),
                            rec.get("time"),
                            rec.get("conf"),
                            rec.get("provenance"),
                            emb_json,
                        )
                    )
        elif fmt == "parquet":
            nodes_df = io.read_parquet(path / "relational_nodes.parquet")
            edges_df = io.read_parquet(path / "relational_edges.parquet")
            for rec in nodes_df.to_dict(orient="records"):
                emb = (
                    np.asarray(rec.get("embedding"), dtype=float)
                    if rec.get("embedding") is not None
                    else None
                )
                self._restore_node((rec["name"], emb, rec.get("node_type")))
            for rec in edges_df.to_dict(orient="records"):
                emb_json = (
                    json.dumps(rec.get("embedding")) if rec.get("embedding") is not None else None
                )
                self._restore_edge(
                    (
                        rec["id"],
                        rec["src"],
                        rec["relation"],
                        rec["dst"],
                        rec.get("context"),
                        rec.get("time"),
                        rec.get("conf"),
                        rec.get("provenance"),
                        emb_json,
                    )
                )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {fmt}")

        self.counters["nodes_added"] = self.graph.number_of_nodes()
        self.counters["edges_added"] = self.graph.number_of_edges()

    def log_status(self) -> dict[str, int]:  # pragma: no cover - simple accessor
        """Return diagnostic counters for this graph."""

        return {**self._log, **self.counters}


__all__ = ["KnowledgeGraph"]
