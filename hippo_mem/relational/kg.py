"""Lightweight knowledge graph implementation.

This module provides a very small wrapper around :mod:`networkx` that also
allows nodes to be associated with embedding vectors.  The
:class:`KnowledgeGraph` class is intentionally minimal but supports adding
tuples extracted by :func:`hippo_mem.relational.tuples.extract_tuples` and
retrieving a relevant subgraph given a query embedding.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import networkx as nx
import numpy as np


class KnowledgeGraph:
    """In-memory knowledge graph with embedding based lookup."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.embeddings: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Graph manipulation
    def add_tuple(
        self, entity: str, relation: str, context: str, time: Optional[str] = None
    ) -> None:
        """Add a tuple to the graph.

        The tuple is represented as an edge from ``entity`` to a node named by
        the free-form ``relation`` string.  ``context`` and ``time`` are stored as
        edge attributes.
        """

        rel_node = relation
        self.graph.add_node(entity)
        self.graph.add_node(rel_node)
        self.graph.add_edge(entity, rel_node, context=context, time=time)

    # ------------------------------------------------------------------
    # Embeddings
    def set_embedding(self, node: str, vector: Iterable[float]) -> None:
        """Associate an embedding vector with ``node``."""

        self.embeddings[node] = np.asarray(list(vector), dtype=float)

    # ------------------------------------------------------------------
    # Retrieval
    def subgraph(self, query: Iterable[float], k: int = 1, radius: int = 1) -> nx.DiGraph:
        """Retrieve an induced subgraph around the top ``k`` nodes.

        Similarity is measured by the dot product between the query vector and
        stored node embeddings.  The returned subgraph is the union of
        ``radius``-hop ego graphs around the selected nodes.
        """

        if not self.embeddings:
            return nx.DiGraph()

        q = np.asarray(list(query), dtype=float)
        scores = {n: float(np.dot(vec, q)) for n, vec in self.embeddings.items()}
        top_nodes = sorted(scores, key=scores.get, reverse=True)[:k]

        nodes = set()
        for node in top_nodes:
            if node in self.graph:
                nodes.update(nx.ego_graph(self.graph, node, radius).nodes())

        return self.graph.subgraph(nodes).copy()


__all__ = ["KnowledgeGraph"]
