"""Shared retrieval utilities for memory modules.

This module consolidates the repeated logic for padding retrieved vectors,
projecting them to the model dimension, assembling metadata and updating
telemetry. Memory-specific retrieval wrappers can delegate to these helpers
for consistent behaviour.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

from hippo_mem.retrieval.telemetry import record_stats

from . import MemoryTokens


def pad_and_project(
    vecs: np.ndarray,
    hits: int,
    limit: int,
    base_dim: int,
    proj: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """Pad ``vecs`` to ``limit`` and project to model dimension."""

    if hits < limit:
        pad = np.zeros((limit - hits, base_dim), dtype="float32")
        vecs = np.vstack([vecs, pad]) if hits else pad
    mask = torch.zeros(limit, dtype=torch.bool, device=device)
    mask[:hits] = True
    vec_t = torch.from_numpy(vecs).to(device=device, dtype=dtype)
    vec_t = proj(vec_t)
    return vec_t, mask


def build_meta(
    kind: str,
    start: float,
    hits: int,
    k: int,
    *,
    bsz: int,
    **extra: Any,
) -> dict[str, float]:
    """Construct metadata dictionary for retrieval calls."""

    latency_ms = (time.perf_counter() - start) * 1000
    hit_rate = hits / (bsz * k) if k > 0 and bsz > 0 else 0.0
    meta = {
        "source": kind,
        "k": k,
        "batch_size": bsz,
        "latency_ms": latency_ms,
        "hit_rate": hit_rate,
    }
    meta.update(extra)
    return meta


def retrieve_and_pack_base(
    retrieve_fn: Callable[[], Iterable[tuple[np.ndarray, int]]],
    *,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
    proj: nn.Module,
    build_meta_fn: Callable[..., dict[str, float]],
    telemetry_key: str,
) -> MemoryTokens:
    """Execute retrieval, pad/project vectors and assemble ``MemoryTokens``."""

    start = time.perf_counter()
    results = list(retrieve_fn())
    bsz = len(results)
    base_dim = getattr(proj, "in_features", results[0][0].shape[1] if results else 0)

    packed = []
    mask_rows = []
    hits_total = 0
    for vecs, hits in results:
        vec_t, mask = pad_and_project(vecs, hits, k, base_dim, proj, device, dtype)
        packed.append(vec_t)
        mask_rows.append(mask)
        hits_total += hits

    d_model = getattr(proj, "out_features", base_dim)
    tokens = (
        torch.stack(packed) if packed else torch.zeros(0, k, d_model, device=device, dtype=dtype)
    )
    mask = (
        torch.stack(mask_rows) if mask_rows else torch.zeros(0, k, dtype=torch.bool, device=device)
    )
    meta = build_meta_fn(start=start, hits=hits_total, k=k, bsz=bsz)

    record_stats(
        telemetry_key,
        k=k,
        batch_size=bsz,
        hits=hits_total,
        tokens=tokens.shape[1],
        latency_ms=meta.get("latency_ms", 0.0),
    )
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)


__all__ = ["pad_and_project", "build_meta", "retrieve_and_pack_base"]
