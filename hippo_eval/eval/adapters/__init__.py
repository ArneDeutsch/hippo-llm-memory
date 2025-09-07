from __future__ import annotations

from typing import Dict

from omegaconf import DictConfig

from .base import EvalAdapter
from .episodic import EpisodicEvalAdapter
from .relational import RelationalEvalAdapter

REGISTRY: Dict[str, EvalAdapter] = {
    "episodic": EpisodicEvalAdapter(),
    "relational": RelationalEvalAdapter(),
}


def enabled_adapters(cfg: DictConfig) -> Dict[str, EvalAdapter]:
    """Return adapters enabled by ``cfg``."""

    mem_cfg = getattr(cfg, "memory", {}) or {}
    active: Dict[str, EvalAdapter] = {}
    if mem_cfg is None:
        return active
    # Support memory specified as list or dict
    if isinstance(mem_cfg, (list, tuple)):
        names = set(mem_cfg)
    else:
        names = set(getattr(mem_cfg, "keys", lambda: mem_cfg.keys())())
    if "episodic" in REGISTRY and ("episodic" in names or "hei_nw" in names):
        active["episodic"] = REGISTRY["episodic"]
    if "relational" in REGISTRY and ("relational" in names or "sgc_rss" in names):
        active["relational"] = REGISTRY["relational"]
    return active


__all__ = [
    "EvalAdapter",
    "EpisodicEvalAdapter",
    "RelationalEvalAdapter",
    "enabled_adapters",
]
