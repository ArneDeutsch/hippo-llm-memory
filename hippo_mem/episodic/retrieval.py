"""Utility for episodic memory retrieval and packing.

Supports optional Hopfield completion of cues before FAISS lookup.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from torch import nn

from hippo_mem.common import MemoryTokens, TraceSpec
from hippo_mem.common.retrieval import build_meta, retrieve_and_pack_base
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
) -> tuple[np.ndarray, int, int]:
    """Optionally apply Hopfield completion to the cue and merge with vectors.

    Returns
    -------
    tuple
        ``(vecs, mask_hits, placeholder)`` where ``mask_hits`` is the number of
        tokens marked as real for packing and ``placeholder`` counts injected
        cues that should **not** be tallied as retrieval hits for telemetry.
    """

    placeholder = 0
    use_hopfield = getattr(spec, "params", {}).get("hopfield", True)
    if k > 0 and use_hopfield and hasattr(store, "complete"):
        cue = store.complete(cue, k=k)
        if hits > 0:
            vecs[0] = cue
            mask_hits = hits
        else:
            vecs = cue.reshape(1, -1)
            mask_hits = 1
            placeholder = 1
        return vecs, mask_hits, placeholder
    return vecs, hits, placeholder


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

    placeholder_total = 0
    actual_hits = 0

    def iter_retrieve():
        nonlocal placeholder_total, actual_hits
        for i in range(bsz):
            cue = batch_hidden[i, -1].detach().cpu().numpy()
            cue, vecs, hits = _recall_traces(store, cue, k)
            vecs, mask_hits, placeholder = _apply_hopfield(store, vecs, cue, hits, k, spec)
            placeholder_total += placeholder
            actual_hits += hits
            yield vecs, mask_hits

    def meta_fn(start: float, hits: int, k: int, bsz: int) -> dict[str, float]:
        return build_meta("episodic", start, hits, k, bsz=bsz)

    mem = retrieve_and_pack_base(
        iter_retrieve,
        k=k,
        device=device,
        dtype=dtype,
        proj=proj,
        build_meta_fn=meta_fn,
        telemetry_key="episodic",
    )

    if placeholder_total:
        stats = registry.get("episodic")
        stats.hits -= placeholder_total
    if k > 0 and bsz > 0:
        mem.meta["hit_rate"] = actual_hits / (bsz * k)

    logger.info(
        "episodic_retrieval_k=%d episodic_latency_ms=%.2f",
        k,
        mem.meta.get("latency_ms", 0.0),
    )
    return mem
