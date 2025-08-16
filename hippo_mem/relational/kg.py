"""Lightweight persistent knowledge graph implementation."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, Optional, Sequence

import networkx as nx
import numpy as np

from .schema import SchemaIndex
from .tuples import TupleType


class KnowledgeGraph:
    """Knowledge graph backed by NetworkX and SQLite."""

    def __init__(self, db_path: str = ":memory:", *, config: Optional[dict] = None) -> None:
        self.graph = nx.MultiDiGraph()
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._load()
        self.config = config or {}
        thresh = float(self.config.get("schema_threshold", 0.8))
        self.schema_index = SchemaIndex(threshold=thresh)
        self._gnn_updates = bool(self.config.get("gnn_updates", True))
        self._log = {"writes": 0, "recalls": 0, "hits": 0, "maintenance": 0}
        self._bg_thread: Optional[threading.Thread] = None
        self._history: list[dict[str, Any]] = []
        self._max_undo = int(self.config.get("max_undo", 5))
        self._maintenance_log: list[dict[str, Any]] = []
        self._log_file = self.config.get("maintenance_log")

    # ------------------------------------------------------------------
    # Database utilities
    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS nodes (name TEXT PRIMARY KEY, embedding TEXT)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                src TEXT,
                relation TEXT,
                dst TEXT,
                context TEXT,
                time TEXT,
                conf REAL,
                provenance INTEGER,
                embedding TEXT
            )
            """
        )
        self.conn.commit()

    def _load(self) -> None:
        cur = self.conn.cursor()
        for name, emb in cur.execute("SELECT name, embedding FROM nodes"):
            self.graph.add_node(name)
            if emb is not None:
                self.node_embeddings[name] = np.asarray(json.loads(emb), dtype=float)
        for edge_id, src, rel, dst, ctx, t, conf, prov, emb in cur.execute(
            "SELECT id, src, relation, dst, context, time, conf, provenance, embedding FROM edges"
        ):
            self.graph.add_edge(
                src,
                dst,
                key=edge_id,
                relation=rel,
                context=ctx,
                time=t,
                conf=conf,
                provenance=prov,
            )
            if emb is not None:
                self.graph[src][dst][edge_id]["embedding"] = np.asarray(
                    json.loads(emb), dtype=float
                )

    # ------------------------------------------------------------------
    # Graph manipulation
    def upsert(
        self,
        head: str,
        relation: str,
        tail: str,
        context: str,
        time: Optional[str] = None,
        conf: float = 1.0,
        provenance: Optional[int] = None,
        *,
        head_embedding: Optional[Iterable[float]] = None,
        tail_embedding: Optional[Iterable[float]] = None,
        edge_embedding: Optional[Iterable[float]] = None,
    ) -> None:
        """Add or update a tuple in the graph and SQLite store."""

        self.graph.add_node(head)
        self.graph.add_node(tail)

        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)",
            (head, self._to_json(head_embedding)),
        )
        cur.execute(
            "INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)",
            (tail, self._to_json(tail_embedding)),
        )
        cur.execute(
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
        )
        edge_id = cur.lastrowid
        self.conn.commit()

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

    def ingest(self, tup: TupleType) -> bool:
        """Route a tuple through ``SchemaIndex`` and insert if confident."""

        return self.schema_index.fast_track(tup, self)

    def _to_json(self, emb: Optional[Iterable[float]]) -> Optional[str]:
        if emb is None:
            return None
        return json.dumps(list(map(float, emb)))

    def _gnn_update(self, nodes: Sequence[str]) -> None:
        """Very small message-passing stub updating node embeddings."""

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
        """Return a radius-``r`` subgraph around the top-``k`` nodes."""

        if not self.node_embeddings:
            return nx.MultiDiGraph()

        q = np.asarray(list(query_embedding), dtype=float)
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

    # ------------------------------------------------------------------
    # Maintenance helpers
    def _log_event(self, op: str, info: dict[str, Any]) -> None:
        event = {"ts": time.time(), "op": op, **info}
        self._maintenance_log.append(event)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")

    def _push_history(self, entry: dict[str, Any]) -> None:
        self._history.append(entry)
        if len(self._history) > self._max_undo:
            self._history.pop(0)

    # ------------------------------------------------------------------
    # Maintenance and logging
    def prune(self, min_conf: float = 0.0, max_age: Optional[float] = None) -> None:
        """Remove edges below confidence or older than ``max_age`` seconds."""

        cur = self.conn.cursor()
        conditions: list[str] = []
        params: list[float] = []
        if min_conf is not None:
            conditions.append("conf < ?")
            params.append(min_conf)
        if max_age is not None:
            cutoff = time.time() - max_age
            conditions.append("(time IS NOT NULL AND CAST(time AS REAL) < ?)")
            params.append(cutoff)
        if not conditions:
            return
        where = " OR ".join(conditions)
        cur.execute(
            f"SELECT id, src, relation, dst, context, time, conf, provenance, embedding FROM edges WHERE {where}",
            params,
        )
        edges = cur.fetchall()
        if not edges:
            return
        candidate_nodes = {e[1] for e in edges} | {e[3] for e in edges}
        node_data = {n: self.node_embeddings.get(n) for n in candidate_nodes}
        for edge_id, src, rel, dst, ctx, t, conf, prov, emb in edges:
            if self.graph.has_edge(src, dst, key=edge_id):
                self.graph.remove_edge(src, dst, key=edge_id)
            cur.execute("DELETE FROM edges WHERE id=?", (edge_id,))
        self.conn.commit()

        removed_nodes: list[tuple[str, Optional[np.ndarray]]] = []
        for n in list(candidate_nodes):
            if self.graph.degree(n) == 0:
                self.graph.remove_node(n)
                cur.execute("DELETE FROM nodes WHERE name=?", (n,))
                removed_nodes.append((n, node_data.get(n)))
                self.node_embeddings.pop(n, None)
        self.conn.commit()
        self._push_history({"op": "prune", "edges": edges, "nodes": removed_nodes})
        self._log_event("prune", {"min_conf": min_conf, "max_age": max_age})

    def log_status(self) -> dict:
        return dict(self._log)

    def start_background_tasks(self, interval: float = 300.0) -> None:
        if self._bg_thread is not None:
            return

        def loop() -> None:
            while True:
                time.sleep(interval)
                cfg = self.config.get("prune", {})
                self.prune(
                    float(cfg.get("min_conf", 0.0)),
                    cfg.get("max_age"),
                )
                self._log["maintenance"] += 1

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._bg_thread = t

    def rollback(self, n: int = 1) -> None:
        """Rollback the last ``n`` prune operations."""

        for _ in range(n):
            if not self._history:
                break
            entry = self._history.pop()
            if entry.get("op") != "prune":
                continue
            edges = entry.get("edges", [])
            nodes = entry.get("nodes", [])
            cur = self.conn.cursor()
            for name, emb in nodes:
                self.graph.add_node(name)
                cur.execute(
                    "INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)",
                    (name, self._to_json(emb)),
                )
                if emb is not None:
                    self.node_embeddings[name] = np.asarray(emb, dtype=float)
            for edge in edges:
                edge_id, src, rel, dst, ctx, t_, conf, prov, emb = edge
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
                    self.graph[src][dst][edge_id]["embedding"] = np.asarray(
                        json.loads(emb), dtype=float
                    )
                cur.execute(
                    "INSERT INTO edges(id, src, relation, dst, context, time, conf, provenance, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
                    (edge_id, src, rel, dst, ctx, t_, conf, prov, emb),
                )
            self.conn.commit()
            self._log_event("rollback", {"op": "prune"})


__all__ = ["KnowledgeGraph"]
