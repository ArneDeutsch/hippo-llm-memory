"""Lightweight persistent knowledge graph implementation."""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, Iterable, Optional

import networkx as nx
import numpy as np


class KnowledgeGraph:
    """Knowledge graph backed by NetworkX and SQLite."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.graph = nx.DiGraph()
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self._load()

    # ------------------------------------------------------------------
    # Database utilities
    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS nodes (name TEXT PRIMARY KEY, embedding TEXT)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                src TEXT,
                dst TEXT,
                context TEXT,
                time TEXT,
                embedding TEXT,
                PRIMARY KEY(src, dst)
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
        for src, dst, ctx, time, emb in cur.execute(
            "SELECT src, dst, context, time, embedding FROM edges"
        ):
            self.graph.add_edge(src, dst, context=ctx, time=time)
            if emb is not None:
                self.graph[src][dst]["embedding"] = np.asarray(json.loads(emb), dtype=float)

    # ------------------------------------------------------------------
    # Graph manipulation
    def upsert(
        self,
        entity: str,
        relation: str,
        context: str,
        time: Optional[str] = None,
        *,
        entity_embedding: Optional[Iterable[float]] = None,
        relation_embedding: Optional[Iterable[float]] = None,
        edge_embedding: Optional[Iterable[float]] = None,
    ) -> None:
        """Add or update a tuple in the graph and SQLite store."""

        rel_node = relation
        self.graph.add_node(entity)
        self.graph.add_node(rel_node)
        self.graph.add_edge(entity, rel_node, context=context, time=time)
        if edge_embedding is not None:
            self.graph[entity][rel_node]["embedding"] = np.asarray(
                list(edge_embedding), dtype=float
            )

        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)",
            (entity, self._to_json(entity_embedding)),
        )
        cur.execute(
            "INSERT OR REPLACE INTO nodes(name, embedding) VALUES (?, ?)",
            (rel_node, self._to_json(relation_embedding)),
        )
        cur.execute(
            "INSERT OR REPLACE INTO edges(src, dst, context, time, embedding) VALUES (?,?,?,?,?)",
            (entity, rel_node, context, time, self._to_json(edge_embedding)),
        )
        self.conn.commit()

        if entity_embedding is not None:
            self.node_embeddings[entity] = np.asarray(list(entity_embedding), dtype=float)
        if relation_embedding is not None:
            self.node_embeddings[rel_node] = np.asarray(list(relation_embedding), dtype=float)

    def _to_json(self, emb: Optional[Iterable[float]]) -> Optional[str]:
        if emb is None:
            return None
        return json.dumps(list(map(float, emb)))

    # ------------------------------------------------------------------
    # Retrieval
    def retrieve(self, query_embedding: Iterable[float], k: int = 1, radius: int = 1) -> nx.DiGraph:
        """Return a radius-``r`` subgraph around the top-``k`` nodes."""

        if not self.node_embeddings:
            return nx.DiGraph()

        q = np.asarray(list(query_embedding), dtype=float)
        scores = {n: float(np.dot(vec, q)) for n, vec in self.node_embeddings.items()}
        top = sorted(scores, key=scores.get, reverse=True)[:k]

        nodes = set()
        for n in top:
            if n in self.graph:
                nodes.update(nx.ego_graph(self.graph, n, radius).nodes())

        return self.graph.subgraph(nodes).copy()


__all__ = ["KnowledgeGraph"]
