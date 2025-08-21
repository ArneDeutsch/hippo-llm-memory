"""Utilities for relational memory retrieval and packing."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import torch
from torch import nn

from hippo_mem.common import MemoryTokens, TraceSpec

from .kg import KnowledgeGraph


def _pool_node(kg: KnowledgeGraph, sub: nx.MultiDiGraph, node: str, dim: int) -> np.ndarray:
    """Average pool ``node`` and its neighbors' embeddings."""

    emb = []
    if node in kg.node_embeddings:
        emb.append(kg.node_embeddings[node])
    for nbr in sub.neighbors(node):
        if nbr in kg.node_embeddings:
            emb.append(kg.node_embeddings[nbr])
    if emb:
        return np.mean(emb, axis=0)
    return np.zeros(dim, dtype=np.float32)


def relational_retrieve_and_pack(
    batch_hidden: torch.FloatTensor,
    spec: TraceSpec,
    kg: KnowledgeGraph,
    proj: nn.Module,
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
    d_model = getattr(proj, "out_features", batch_hidden.size(-1))
    start = time.perf_counter()

    base_dim = kg.dim or batch_hidden.size(-1)
    packed = []
    mask = torch.zeros(bsz, limit, dtype=torch.bool, device=device)
    hit_total = 0

    for i in range(bsz):
        query = batch_hidden[i, -1].detach().cpu().numpy()
        sub = kg.retrieve(query, k=seeds, radius=hops) if limit > 0 else nx.MultiDiGraph()
        nodes = list(sub.nodes())[:limit]
        hits = len(nodes)
        hit_total += hits
        vecs = [_pool_node(kg, sub, n, base_dim) for n in nodes]
        if hits < limit:
            vecs.extend([np.zeros(base_dim, dtype=np.float32) for _ in range(limit - hits)])
        mask[i, :hits] = True
        arr = np.stack(vecs) if vecs else np.zeros((limit, base_dim), dtype=np.float32)
        vec_t = torch.from_numpy(arr).to(device=device, dtype=dtype)
        vec_t = proj(vec_t)
        packed.append(vec_t)

    tokens = torch.stack(packed) if packed else torch.zeros(0, limit, d_model, device=device)
    latency_ms = (time.perf_counter() - start) * 1000
    hit_rate = hit_total / (bsz * limit) if limit > 0 else 0.0
    meta = {
        "source": "relational",
        "k": limit,
        "hops": hops,
        "seeds": seeds,
        "latency_ms": latency_ms,
        "hit_rate": hit_rate,
    }
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)


__all__ = ["relational_retrieve_and_pack"]
