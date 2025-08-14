"""Adapter for relational memory."""

from __future__ import annotations

import numpy as np


class RelationalAdapter:
    """Stub adapter bridging model and knowledge graph."""

    def __call__(self, query: np.ndarray, graph_feats: np.ndarray) -> np.ndarray:
        """Fuse ``query`` with ``graph_feats`` via simple cross-attention.

        Args:
            query: Feature vector representing the query.
            graph_feats: Matrix of node features for an encoded subgraph.

        Returns:
            A single feature vector computed as the attention weighted sum of
            ``graph_feats``.
        """

        q = np.asarray(query, dtype=float)
        g = np.asarray(graph_feats, dtype=float)
        if g.ndim == 1:
            g = g[None, :]
        scores = g @ q
        weights = np.exp(scores - scores.max())
        weights /= weights.sum() if weights.size else 1.0
        return weights @ g


__all__ = ["RelationalAdapter"]
