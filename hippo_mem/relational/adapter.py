"""Dual-path adapter for semantic and episodic fusion.

Summary
-------
Implements deterministic fusion of semantic graph features and episodic
trace features. Each path uses a simple attention mechanism, and outputs
are gated by confidence scores.

Parameters
----------
None

Returns
-------
None

Raises
------
None

Side Effects
------------
None

Complexity
----------
Attention is ``O(ND)`` for ``N`` features of dimension ``D``.

Examples
--------
>>> adapter = RelationalAdapter()
>>> q = np.array([1.0, 0.0])
>>> g = np.array([[1.0, 0.0]])
>>> adapter(q, g)
array([1., 0.])

See Also
--------
hippo_mem.relational.kg.KnowledgeGraph
"""

from __future__ import annotations

import numpy as np


class RelationalAdapter:
    """Fuse features from semantic graph and episodic traces.

    Summary
    -------
    Uses cross-attention on each source and merges them deterministically
    according to provided confidences.
    """

    def _attend(self, query: np.ndarray, feats: np.ndarray) -> np.ndarray:
        """Scaled dot-product attention over ``feats``.

        Parameters
        ----------
        query : np.ndarray
            Query vector of shape ``(D,)``.
        feats : np.ndarray
            Feature matrix of shape ``(N, D)``.

        Returns
        -------
        np.ndarray
            Weighted sum with shape ``(D,)``.
        """

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
        """Fuse ``query`` with semantic and episodic features.

        Summary
        -------
        Applies attention to each feature set then linearly combines the
        results using normalized confidences.

        Parameters
        ----------
        query : np.ndarray
            Query vector ``(D,)``.
        kg_feats : np.ndarray
            Semantic graph features ``(N_k, D)``.
        episodic_feats : np.ndarray, optional
            Episodic features ``(N_e, D)``.
        kg_conf : float, optional
            Confidence weight for KG path.
        episodic_conf : float, optional
            Confidence weight for episodic path.

        Returns
        -------
        np.ndarray
            Fused representation ``(D,)``.

        Side Effects
        ------------
        None

        Complexity
        ----------
        ``O((N_k + N_e) D)``.

        Examples
        --------
        >>> adapter = RelationalAdapter()
        >>> q = np.array([1.0, 0.0])
        >>> g = np.array([[1.0, 0.0]])
        >>> e = np.array([[0.0, 1.0]])
        >>> adapter(q, g, e, kg_conf=1.0, episodic_conf=0.0)
        array([1., 0.])

        See Also
        --------
        hippo_mem.relational.tuples.extract_tuples
        """

        kg_vec = self._attend(query, kg_feats)
        epi_vec = (
            self._attend(query, episodic_feats)
            if episodic_feats is not None and episodic_feats.size
            else np.zeros_like(kg_vec)
        )
        denom = kg_conf + episodic_conf
        # why: gating ensures deterministic fusion based on source confidence
        gate = kg_conf / denom if denom else 0.5
        return gate * kg_vec + (1.0 - gate) * epi_vec


__all__ = ["RelationalAdapter"]
