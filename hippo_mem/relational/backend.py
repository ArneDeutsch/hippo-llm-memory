from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, Literal, Optional, Protocol

import networkx as nx
import numpy as np

from hippo_mem.common.sqlite import SQLiteExecMixin

Fetch = Optional[Literal["one", "all", "lastrowid"]]


class PersistenceStrategy(Protocol):
    """Interface for graph persistence backends."""

    def init_db(self) -> None:
        """Ensure underlying storage is initialized."""

    def load(self, graph: nx.MultiDiGraph, node_embeddings: Dict[str, np.ndarray]) -> None:
        """Populate ``graph`` and ``node_embeddings`` from storage."""

    def exec(self, sql: str, params: Iterable = (), *, fetch: Fetch = None) -> Any:
        """Execute ``sql`` with ``params`` and optionally fetch results."""


class SQLiteBackend(SQLiteExecMixin):
    """SQLite-backed persistence strategy."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def init_db(self) -> None:
        self.exec("CREATE TABLE IF NOT EXISTS nodes (name TEXT PRIMARY KEY, embedding TEXT)")
        self.exec(
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

    def load(self, graph: nx.MultiDiGraph, node_embeddings: Dict[str, np.ndarray]) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT name, embedding FROM nodes")
        for name, emb_json in cur.fetchall():
            graph.add_node(name)
            if emb_json:
                node_embeddings[name] = np.asarray(json.loads(emb_json), dtype=float)
        cur.execute(
            "SELECT id, src, relation, dst, context, time, conf, provenance, embedding FROM edges"
        )
        for row in cur.fetchall():
            emb = np.asarray(json.loads(row[8]), dtype=float) if row[8] else None
            graph.add_edge(
                row[1],
                row[3],
                key=row[0],
                relation=row[2],
                context=row[4],
                time=row[5],
                conf=row[6],
                provenance=row[7],
                embedding=emb,
            )

    def exec(self, sql: str, params: Iterable = (), *, fetch: Fetch = None) -> Any:
        cur = self.conn.cursor()
        cur.execute(sql, params)
        result: Any = None
        if fetch == "one":
            result = cur.fetchone()
        elif fetch == "all":
            result = cur.fetchall()
        elif fetch == "lastrowid":
            result = cur.lastrowid
        self.conn.commit()
        return result


__all__ = ["PersistenceStrategy", "SQLiteBackend"]
