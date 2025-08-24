"""Utility for episodic memory retrieval and packing.

Supports optional Hopfield completion of cues before FAISS lookup.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import nn

from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.common.retrieval import build_meta, retrieve_and_pack_base

from .store import EpisodicStore


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




def episodic_retrieve_and_pack(
    batch_hidden: torch.FloatTensor,
    spec: TraceSpec,
    store: EpisodicStore,
    proj: nn.Module,
) -> MemoryTokens:
    """Recall episodic traces and package them as ``MemoryTokens``."""

    k = spec.k or 0
    device = batch_hidden.device
    dtype = batch_hidden.dtype
    bsz = batch_hidden.size(0)

    def iterator():
        for i in range(bsz):
            cue = batch_hidden[i, -1].detach().cpu().numpy()
            cue, vecs, hits = _recall_traces(store, cue, k)
            vecs, hits = _apply_hopfield(store, vecs, cue, hits, k, spec)
            yield vecs, hits, store.dim

    def meta_fn(**kw):
        return build_meta("episodic", **kw)

    tokens = retrieve_and_pack_base(
        iterator,
        k=k,
        device=device,
        dtype=dtype,
        proj=proj,
        build_meta_fn=meta_fn,
        telemetry_key="episodic",
    )
    return tokens
