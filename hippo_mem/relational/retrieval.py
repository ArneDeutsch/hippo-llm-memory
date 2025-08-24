"""Utilities for relational memory retrieval and packing."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch import nn

from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.common.retrieval import build_meta, retrieve_and_pack_base

from .kg import KnowledgeGraph


@dataclass
class NodePooler:
    """Pool node embeddings according to a policy."""

    policy: str = "mean"

    def __call__(
        self,
        kg: KnowledgeGraph,
        sub: nx.MultiDiGraph,
        node: str,
        dim: int,
    ) -> np.ndarray:
        """Return a pooled embedding for ``node``."""

        emb: list[np.ndarray] = []
        if self.policy == "mean":
            if node in kg.node_embeddings:
                emb.append(kg.node_embeddings[node])
            for nbr in sub.neighbors(node):
                if nbr in kg.node_embeddings:
                    emb.append(kg.node_embeddings[nbr])
            if emb:
                return np.mean(emb, axis=0)
        elif self.policy == "self":
            if node in kg.node_embeddings:
                return kg.node_embeddings[node]
        else:  # pre: policy supported
            raise ValueError(f"unknown policy {self.policy}")
        return np.zeros(dim, dtype=np.float32)


def _retrieve_subgraph(
    kg: KnowledgeGraph, query: np.ndarray, seeds: int, hops: int, limit: int
) -> tuple[nx.MultiDiGraph, list[str]]:
    """Retrieve a neighborhood subgraph and node list."""

    if limit <= 0:
        return nx.MultiDiGraph(), []
    sub = kg.retrieve(query, k=seeds, radius=hops)
    nodes = list(sub.nodes())[:limit]
    return sub, nodes




def relational_retrieve_and_pack(
    batch_hidden: torch.FloatTensor,
    spec: TraceSpec,
    kg: KnowledgeGraph,
    proj: nn.Module,
    pooler: NodePooler | None = None,
) -> MemoryTokens:
    """Retrieve KG neighborhoods and package as ``MemoryTokens``.

    Parameters
    ----------
    batch_hidden:
        Final hidden states ``[B, T, H]`` from the model.
    spec:
        Retrieval specification with ``k`` limit and ``params['hops']``.
    kg:
        Knowledge graph providing ``retrieve`` and node embeddings.
    proj:
        Module projecting KG embeddings to model dimension.
    pooler:
        Strategy aggregating node embeddings.

    Returns
    -------
    MemoryTokens
        Packed tokens ``[B, k, d_model]`` and mask.
    """

    limit = int(spec.k or spec.params.get("limit", 0) or 0)
    hops = int(spec.params.get("hops", 1))
    seeds = int(spec.params.get("seeds", 1))
    if limit > 0:
        seeds = min(seeds, limit)
    bsz = batch_hidden.size(0)
    device = batch_hidden.device
    dtype = batch_hidden.dtype
    pooler = pooler or NodePooler()
    base_dim = kg.dim or batch_hidden.size(-1)

    def iterator():
        for i in range(bsz):
            query = batch_hidden[i, -1].detach().cpu().numpy()
            sub, nodes = _retrieve_subgraph(kg, query, seeds, hops, limit)
            vecs = [pooler(kg, sub, n, base_dim) for n in nodes]
            arr = np.stack(vecs) if vecs else np.zeros((0, base_dim), dtype=np.float32)
            yield arr, len(nodes), base_dim

    def meta_fn(**kw):
        return build_meta("relational", **kw, hops=hops, seeds=seeds)

    return retrieve_and_pack_base(
        iterator,
        k=limit,
        device=device,
        dtype=dtype,
        proj=proj,
        build_meta_fn=meta_fn,
        telemetry_key="relational",
    )


__all__ = ["NodePooler", "relational_retrieve_and_pack"]
