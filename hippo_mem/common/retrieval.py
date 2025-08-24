from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn

from . import MemoryTokens
from .telemetry import registry


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
        pad = np.zeros((limit - hits, base_dim), dtype=np.float32)
        vecs = np.vstack([vecs, pad]) if hits else pad
    mask = torch.zeros(limit, dtype=torch.bool, device=device)
    if hits > 0:
        mask[:hits] = True
    vec_t = torch.from_numpy(vecs).to(device=device, dtype=dtype)
    vec_t = proj(vec_t)
    return vec_t, mask


def build_meta(
    kind: str,
    start: float,
    hits: int,
    k: int,
    **extra,
) -> dict[str, float]:
    """Construct a metadata dictionary for retrieval operations."""

    bsz = extra.pop("bsz", 1)
    latency_ms = (time.perf_counter() - start) * 1000
    hit_rate = hits / (bsz * k) if k > 0 and bsz > 0 else 0.0
    return {"source": kind, "k": k, "latency_ms": latency_ms, "hit_rate": hit_rate, **extra}


def retrieve_and_pack_base(
    retrieve_fn: Callable[[], Iterable[tuple[np.ndarray, int, int]]],
    *,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
    proj: nn.Module,
    build_meta_fn,
    telemetry_key: str,
) -> MemoryTokens:
    """Generic retrieval loop producing ``MemoryTokens``."""

    start = time.perf_counter()
    packed: list[torch.Tensor] = []
    masks: list[torch.BoolTensor] = []
    hit_total = 0
    bsz = 0
    for vecs, hits, base_dim in retrieve_fn():
        vec_t, mask_row = pad_and_project(vecs, hits, k, base_dim, proj, device, dtype)
        packed.append(vec_t)
        masks.append(mask_row)
        hit_total += hits
        bsz += 1
    d_model = getattr(proj, "out_features", 0)
    tokens = (
        torch.stack(packed)
        if packed
        else torch.zeros(0, k, d_model, device=device, dtype=dtype)
    )
    mask = (
        torch.stack(masks)
        if masks
        else torch.zeros(0, k, dtype=torch.bool, device=device)
    )
    meta = build_meta_fn(start=start, hits=hit_total, k=k, bsz=bsz)
    k_req = k * bsz
    tokens_returned = int(tokens.shape[1]) if tokens.ndim >= 2 else 0
    latency_ms = meta.get("latency_ms", 0.0)
    registry.get(telemetry_key).update(
        k=k_req, hits=hit_total, tokens=tokens_returned, latency_ms=latency_ms
    )
    return MemoryTokens(tokens=tokens, mask=mask, meta=meta)


__all__ = ["pad_and_project", "build_meta", "retrieve_and_pack_base"]
