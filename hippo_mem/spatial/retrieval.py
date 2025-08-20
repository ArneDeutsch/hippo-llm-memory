"""Spatial subgraph retrieval and packing utilities."""

from __future__ import annotations

import time

import torch
from torch import nn

from hippo_mem.common import MemoryTokens, TraceSpec

from .place_graph import PlaceGraph


def spatial_retrieve_and_pack(
    center: str,
    spec: TraceSpec,
    graph: PlaceGraph,
    proj: nn.Linear,
) -> MemoryTokens:
    """Gather a local subgraph and project to ``d_model`` memory tokens.

    Parameters
    ----------
    center:
        Context name acting as the subgraph origin.
    spec:
        Retrieval configuration.  ``params`` may contain ``radius``,
        ``max_nodes`` and ``max_edges``.
    graph:
        ``PlaceGraph`` providing neighbourhood information.
    proj:
        Linear layer projecting 4-D features to ``d_model``.

    Returns
    -------
    MemoryTokens
        Packed tokens ``[1, M, d_model]`` and mask ``[1, M]`` where ``M`` is
        ``max_nodes + max_edges``.
    """

    radius = int(spec.params.get("radius", 1))
    max_nodes = int(spec.params.get("max_nodes", 8))
    max_edges = int(spec.params.get("max_edges", 8))
    total = max_nodes + max_edges

    device = next(proj.parameters()).device
    dtype = next(proj.parameters()).dtype

    start_t = time.perf_counter()
    nodes, edges = graph.local(center, radius)
    nodes = nodes[:max_nodes]
    edges = edges[:max_edges]

    tokens = torch.zeros(1, total, proj.out_features, device=device, dtype=dtype)
    mask = torch.zeros(1, total, dtype=torch.bool, device=device)

    idx = 0
    for n in nodes:
        ctx = graph._id_to_context[n]
        p = graph.encoder.encode(ctx)
        feat = torch.tensor(
            [p.coord[0], p.coord[1], float(p.last_seen), 1.0], device=device, dtype=dtype
        )
        tokens[0, idx] = proj(feat.unsqueeze(0)).squeeze(0)
        mask[0, idx] = True
        idx += 1
    for u, v in edges:
        e = graph.graph[u][v]
        feat = torch.tensor(
            [e.cost, e.success, float(e.last_seen), 0.0], device=device, dtype=dtype
        )
        tokens[0, idx] = proj(feat.unsqueeze(0)).squeeze(0)
        mask[0, idx] = True
        idx += 1

    latency_ms = (time.perf_counter() - start_t) * 1000
    meta = {
        "radius": radius,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "latency_ms": latency_ms,
    }
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)


__all__ = ["spatial_retrieve_and_pack"]
