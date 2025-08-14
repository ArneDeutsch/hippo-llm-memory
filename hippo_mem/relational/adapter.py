"""Adapter for relational memory."""

from __future__ import annotations

import numpy as np


class RelationalAdapter:
    """Stub adapter bridging model, KG and episodic memory."""

    def _attend(self, query: np.ndarray, feats: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=float)
        g = np.asarray(feats, dtype=float)
        if g.ndim == 1:
            g = g[None, :]
        if g.size == 0:
            return np.zeros_like(q)
        scores = g @ q
        weights = np.exp(scores - scores.max())
        weights /= weights.sum() if weights.size else 1.0
        return weights @ g

    def __call__(
        self,
        query: np.ndarray,
        kg_feats: np.ndarray,
        episodic_feats: np.ndarray | None = None,
        kg_conf: float = 1.0,
        episodic_conf: float = 1.0,
    ) -> np.ndarray:
        """Fuse ``query`` with KG and episodic features via dual cross-attention."""

        kg_vec = self._attend(query, kg_feats)
        epi_vec = (
            self._attend(query, episodic_feats)
            if episodic_feats is not None and episodic_feats.size
            else np.zeros_like(kg_vec)
        )
        denom = kg_conf + episodic_conf
        gate = kg_conf / denom if denom else 0.5
        return gate * kg_vec + (1.0 - gate) * epi_vec


__all__ = ["RelationalAdapter"]
