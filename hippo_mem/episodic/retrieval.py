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
from hippo_mem.common.telemetry import registry

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


def _recall_traces(
    store: EpisodicStore, cue: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Recall ``k`` traces from ``store`` and return their vectors.

    Returns
    -------
    tuple
        Updated cue, dense vectors for recalled traces, and number of hits.
    """

    k_wta = getattr(store, "k_wta", 0)
    if k_wta > 0:
        cue_sparse = store.sparse_encode(cue, k_wta)
        cue = store.to_dense(cue_sparse)
    traces = store.recall(cue, k) if k > 0 else []
    hits = len(traces)
    vecs = _extract_vectors(store, traces, store.dim)
    return cue, vecs, hits


def _apply_hopfield(
    store: EpisodicStore,
    vecs: np.ndarray,
    cue: np.ndarray,
    hits: int,
    k: int,
    spec: TraceSpec,
) -> tuple[np.ndarray, int]:
    """Optionally apply Hopfield completion to the cue and merge with vectors."""

    use_hopfield = getattr(spec, "params", {}).get("hopfield", True)
    if k > 0 and use_hopfield and hasattr(store, "complete"):
        cue = store.complete(cue, k=k)
        if hits > 0:
            vecs[0] = cue
        else:
            vecs = cue.reshape(1, -1)
            hits = 1
    return vecs, hits


def _pad_and_pack(
    vecs: np.ndarray,
    hits: int,
    k: int,
    store_dim: int,
    proj: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """Pad ``vecs`` to ``k`` and project to model dimension."""

    if hits < k:
        pad = np.zeros((k - hits, vecs.shape[1] if hits else store_dim), dtype="float32")
        vecs = np.vstack([vecs, pad]) if hits else pad
    mask = torch.zeros(k, dtype=torch.bool, device=device)
    mask[:hits] = True
    vec_t = torch.from_numpy(vecs).to(device=device, dtype=dtype)
    vec_t = proj(vec_t)
    return vec_t, mask


def episodic_retrieve_and_pack(
    batch_hidden: torch.FloatTensor,
    spec: TraceSpec,
    store: EpisodicStore,
    proj: nn.Module,
) -> MemoryTokens:
    """Recall episodic traces and package them as ``MemoryTokens``."""

    k = spec.k or 0
    bsz = batch_hidden.size(0)
    device = batch_hidden.device
    dtype = batch_hidden.dtype
    d_model = getattr(proj, "out_features", batch_hidden.size(-1))
    start = time.perf_counter()

    if k == 0:
        tokens = torch.zeros(bsz, 0, d_model, device=device)
        mask = torch.zeros(bsz, 0, dtype=torch.bool, device=device)
    else:
        packed = []
        mask_rows = []
        for i in range(bsz):
            cue = batch_hidden[i, -1].detach().cpu().numpy()
            cue, vecs, hits = _recall_traces(store, cue, k)
            vecs, hits = _apply_hopfield(store, vecs, cue, hits, k, spec)
            vec_t, mask_row = _pad_and_pack(vecs, hits, k, store.dim, proj, device, dtype)
            packed.append(vec_t)
            mask_rows.append(mask_row)
        tokens = torch.stack(packed)
        mask = torch.stack(mask_rows)

    latency_ms = (time.perf_counter() - start) * 1000
    hit_rate = mask.sum().item() / (bsz * k) if k > 0 else 0.0
    meta = {"k": k, "latency_ms": latency_ms, "hit_rate": hit_rate}

    k_req = int(k) * bsz
    hits_actual = int(mask.sum().item())
    tokens_returned = int(tokens.shape[1])
    registry.get("episodic").update(
        k=k_req, hits=hits_actual, tokens=tokens_returned, latency_ms=latency_ms
    )

    logger.info("episodic_retrieval_k=%d episodic_latency_ms=%.2f", k, latency_ms)
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)
