"""Specification dataclasses for memory payloads."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch


def _debug() -> bool:
    """Return True when shape validations should run."""
    return os.environ.get("HIPPO_DEBUG", "").lower() in {"1", "true", "yes"}


@dataclass
class MemoryTokens:
    """Container for memory tokens passed to adapters.

    Attributes
    ----------
    tokens : torch.FloatTensor
        Token embeddings ``(B, M, d_model)``.
    mask : torch.BoolTensor
        Valid positions ``(B, M)`` where ``True`` indicates a real token.
    meta : dict[str, Any]
        Optional metadata such as ``source`` or retrieval stats.
    """

    tokens: torch.FloatTensor
    mask: torch.BoolTensor
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - simple shape guards
        if not _debug():
            return
        if self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, M, d_model]")
        if self.mask.ndim != 2:
            raise ValueError("mask must have shape [B, M]")
        if self.tokens.shape[:2] != self.mask.shape:
            raise ValueError("tokens and mask dimensions must align")
        if self.tokens.dtype != torch.float32:
            raise TypeError("tokens must be float32")
        if self.mask.dtype != torch.bool:
            raise TypeError("mask must be bool")


@dataclass
class TraceSpec:
    """Specification for memory retrieval requests."""

    source: str  # kept: used for provenance in reports
    k: int | None = None
    max_len: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - simple value guards
        if not _debug():
            return
        if self.k is not None and self.k < 0:
            raise ValueError("k must be non-negative")
        if self.max_len is not None and self.max_len <= 0:
            raise ValueError("max_len must be positive")
