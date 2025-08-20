"""Utility for episodic memory retrieval and packing.

Supports optional Hopfield completion of cues before FAISS lookup.
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np
import torch
from torch import nn

from hippo_mem.common import MemoryTokens, TraceSpec

from .store import EpisodicStore

logger = logging.getLogger(__name__)


def _extract_vectors(store: EpisodicStore, traces: List, dim: int) -> np.ndarray:
    """Return dense vectors for recalled traces.

    Parameters
    ----------
    store:
        Episodic store used for ``to_dense``.
    traces:
        Sequence returned by :meth:`EpisodicStore.recall`.
    dim:
        Target dimensionality.
    """

    if not traces:
        return np.zeros((0, dim), dtype="float32")
    first = traces[0]
    if isinstance(first, np.ndarray):
        return np.stack(traces)
    return np.stack([store.to_dense(t.key) for t in traces])


def episodic_retrieve_and_pack(
    batch_hidden: torch.FloatTensor,
    spec: TraceSpec,
    store: EpisodicStore,
    proj: nn.Module,
) -> MemoryTokens:
    """Recall episodic traces and package them as ``MemoryTokens``.

    Parameters
    ----------
    batch_hidden:
        Final hidden states ``[B, T, H]`` from the model.
    spec:
        Retrieval specification containing ``k``.
    store:
        Episodic memory store providing ``recall``.
    proj:
        Module projecting store vectors to model dimension.

    Returns
    -------
    MemoryTokens
        Projected memory tokens ``[B, k, d_model]`` and mask.
    """

    k = spec.k or 0
    bsz = batch_hidden.size(0)
    device = batch_hidden.device
    dtype = batch_hidden.dtype
    d_model = getattr(proj, "out_features", batch_hidden.size(-1))
    start = time.perf_counter()

    packed = []
    mask = torch.zeros(bsz, k, dtype=torch.bool, device=device)
    for i in range(bsz):
        cue = batch_hidden[i, -1].detach().cpu().numpy()
        k_wta = getattr(store, "k_wta", 0)
        if k_wta > 0:
            cue = store.to_dense(store.sparse_encode(cue, k_wta))
        traces = store.recall(cue, k) if k > 0 else []
        hits = len(traces)
        vecs = _extract_vectors(store, traces, store.dim)
        use_completion = getattr(spec, "params", {}).get("use_completion", True)
        if k > 0 and use_completion and hasattr(store, "complete"):
            cue = store.complete(cue, k=k)
            if hits > 0:
                vecs[0] = cue
            else:
                vecs = cue.reshape(1, -1)
                hits = 1
        if hits < k:
            pad = np.zeros((k - hits, vecs.shape[1] if hits else store.dim), dtype="float32")
            vecs = np.vstack([vecs, pad]) if hits else pad
        mask[i, :hits] = True
        vec_t = torch.from_numpy(vecs).to(device=device, dtype=dtype)
        vec_t = proj(vec_t)
        packed.append(vec_t)

    tokens = torch.stack(packed) if packed else torch.zeros(0, k, d_model, device=device)
    latency_ms = (time.perf_counter() - start) * 1000
    hit_rate = mask.sum().item() / (bsz * k) if k > 0 else 0.0
    meta = {"k": k, "latency_ms": latency_ms, "hit_rate": hit_rate}
    logger.info(
        "episodic_retrieval_k=%d episodic_latency_ms=%.2f",
        k,
        latency_ms,
    )
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)
