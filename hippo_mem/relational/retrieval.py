"""Utilities for relational memory retrieval and packing."""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch import nn

from hippo_mem.common import MemoryTokens, TraceSpec

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


def _pack_vectors(
    nodes: list[str],
    sub: nx.MultiDiGraph,
    kg: KnowledgeGraph,
    pooler: NodePooler,
    base_dim: int,
    limit: int,
    device: torch.device,
    dtype: torch.dtype,
    proj: nn.Module,
) -> tuple[torch.Tensor, torch.BoolTensor, int]:
    """Pool, pad and project node vectors."""

    hits = len(nodes)
    vecs = [pooler(kg, sub, n, base_dim) for n in nodes]
    if hits < limit:
        vecs.extend([np.zeros(base_dim, dtype=np.float32) for _ in range(limit - hits)])
    arr = np.stack(vecs) if vecs else np.zeros((limit, base_dim), dtype=np.float32)
    vec_t = torch.from_numpy(arr).to(device=device, dtype=dtype)
    vec_t = proj(vec_t)
    mask_row = torch.zeros(limit, dtype=torch.bool, device=device)
    mask_row[:hits] = True
    return vec_t, mask_row, hits


def build_meta(
    limit: int,
    hops: int,
    seeds: int,
    hit_total: int,
    bsz: int,
    start: float,
) -> dict[str, float]:
    """Assemble metadata for ``MemoryTokens``."""

    latency_ms = (time.perf_counter() - start) * 1000
    hit_rate = hit_total / (bsz * limit) if limit > 0 and bsz > 0 else 0.0
    return {
        "source": "relational",
        "k": limit,
        "hops": hops,
        "seeds": seeds,
        "latency_ms": latency_ms,
        "hit_rate": hit_rate,
    }


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
    d_model = getattr(proj, "out_features", batch_hidden.size(-1))
    pooler = pooler or NodePooler()
    start = time.perf_counter()

    base_dim = kg.dim or batch_hidden.size(-1)
    packed: list[torch.Tensor] = []
    mask = torch.zeros(bsz, limit, dtype=torch.bool, device=device)
    hit_total = 0

    for i in range(bsz):
        query = batch_hidden[i, -1].detach().cpu().numpy()
        sub, nodes = _retrieve_subgraph(kg, query, seeds, hops, limit)
        vec_t, mask_row, hits = _pack_vectors(
            nodes, sub, kg, pooler, base_dim, limit, device, dtype, proj
        )
        packed.append(vec_t)
        mask[i] = mask_row
        hit_total += hits

    tokens = torch.stack(packed) if packed else torch.zeros(0, limit, d_model, device=device)
    meta = build_meta(limit, hops, seeds, hit_total, bsz, start)
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)


__all__ = ["NodePooler", "relational_retrieve_and_pack"]
