# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

from omegaconf import DictConfig
from torch import Tensor

from hippo_mem.common import MemoryTokens
from hippo_mem.common.gates import GateCounters


class EvalAdapter(Protocol):
    """Protocol for evaluation memory adapters."""

    def present(self) -> str:
        """Return the memory name presented by the adapter."""

    def build(self, cfg: DictConfig) -> Dict[str, object]:
        """Materialise module objects required for evaluation."""

    def retrieve(
        self,
        cfg: DictConfig,
        modules: Dict[str, object],
        item: object,
        *,
        context_key: str,
        hidden: Tensor,
    ) -> MemoryTokens:
        """Retrieve memory traces for ``item``."""

    def teach(
        self,
        cfg: DictConfig,
        modules: Dict[str, object],
        item: object,
        *,
        dry_run: bool,
        gc: GateCounters,
        suite: str,
    ) -> None:
        """Possibly ingest ``item`` into memory depending on gate decisions."""

    def store_size(self, modules: Dict[str, object]) -> Tuple[int, Dict[str, int]]:
        """Return number of persisted items and diagnostics."""


@dataclass
class RetrieveResult:
    mem: MemoryTokens
    hit: bool
    latency_ms: float
    topk_keys: list[str]
