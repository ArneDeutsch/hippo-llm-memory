"""Spatial subgraph retrieval and packing utilities."""

from __future__ import annotations

import numpy as np
from torch import nn

from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.common.retrieval import build_meta, retrieve_and_pack_base

from .place_graph import PlaceGraph


def spatial_retrieve_and_pack(
    center: str,
    spec: TraceSpec,
    graph: PlaceGraph,
    proj: nn.Linear,
) -> MemoryTokens:
    """Gather a local subgraph and project to ``d_model`` memory tokens.

    Nodes use features ``[x, y, last_seen, 1.0]`` and edges
    ``[cost, success, last_seen, 0.0]`` before projection.  The sequence
    packs all nodes first, then edges.

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

    param = next(proj.parameters())
    device = param.device
    dtype = param.dtype

    nodes, edges = graph.local(center, radius)
    nodes = nodes[:max_nodes]
    edges = edges[:max_edges]
    num_nodes = len(nodes)
    num_edges = len(edges)

    def iterator():
        vecs = []
        for n in nodes:
            ctx = graph._id_to_context[n]
            p = graph.encoder.encode(ctx)
            vecs.append(
                np.array([p.coord[0], p.coord[1], float(p.last_seen), 1.0], dtype=np.float32)
            )
        for u, v in edges:
            e = graph.graph[u][v]
            vecs.append(
                np.array([e.cost, e.success, float(e.last_seen), 0.0], dtype=np.float32)
            )
        arr = np.stack(vecs) if vecs else np.zeros((0, 4), dtype=np.float32)
        yield arr, len(vecs), 4

    def meta_fn(**kw):
        return build_meta(
            "spatial", **kw, radius=radius, num_nodes=num_nodes, num_edges=num_edges
        )

    return retrieve_and_pack_base(
        iterator,
        k=total,
        device=device,
        dtype=dtype,
        proj=proj,
        build_meta_fn=meta_fn,
        telemetry_key="spatial",
    )


__all__ = ["spatial_retrieve_and_pack"]
