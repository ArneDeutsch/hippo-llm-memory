"""Utilities for preprocessing evaluation configs.

These helpers fold shortcut memory blocks into ``cfg.memory`` and apply
ablation flags. They mutate the passed ``DictConfig`` in place.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf, open_dict

from hippo_mem.utils.stores import is_memory_preset


def merge_memory_shortcuts(cfg: DictConfig) -> None:
    """Fold top-level memory blocks into ``cfg.memory``.

    CLI overrides may specify ``episodic.*`` or ``relational.*`` directly for
    convenience. Hydra requires these keys to exist in the schema, so
    ``configs/eval/default.yaml`` defines them as empty dictionaries. This
    helper merges such top-level sections into ``cfg.memory`` and removes the
    shortcuts so downstream code only sees ``cfg.memory``.
    """

    if not isinstance(cfg, DictConfig):
        return
    mem = cfg.get("memory")
    memory_preset = is_memory_preset(str(cfg.get("preset")))
    with open_dict(cfg):
        if not (memory_preset or mem not in (None, {})):
            for name in ("episodic", "relational", "spatial"):
                if name in cfg:
                    cfg.pop(name)
            return
        if mem is None:
            cfg.memory = {}
        mem = cfg.memory
        for name in ("episodic", "relational", "spatial"):
            if name in cfg:
                block = cfg.pop(name)
                if block is None:
                    continue
                with open_dict(mem):
                    if (memory_preset and name in mem) or not memory_preset:
                        mem[name] = OmegaConf.merge(mem.get(name, {}), block)


def apply_ablation_flags(cfg: DictConfig, flat_ablate: dict[str, object]) -> None:
    """Apply flattened ablation flags to ``cfg.memory`` in place."""

    mem_cfg = cfg.get("memory")
    if not isinstance(mem_cfg, DictConfig):
        return
    with open_dict(mem_cfg):
        epi_cfg = mem_cfg.get("episodic")
        if isinstance(epi_cfg, DictConfig):
            if "episodic.use_gate" in flat_ablate:
                gate_cfg = epi_cfg.get("gate")
                if isinstance(gate_cfg, DictConfig):
                    with open_dict(gate_cfg):
                        gate_cfg.enabled = bool(flat_ablate["episodic.use_gate"])
            if "episodic.use_completion" in flat_ablate:
                epi_cfg["use_completion"] = bool(flat_ablate["episodic.use_completion"])
        rel_cfg = mem_cfg.get("relational")
        if isinstance(rel_cfg, DictConfig) and "relational.gate.enabled" in flat_ablate:
            gate_cfg = rel_cfg.get("gate")
            if isinstance(gate_cfg, DictConfig):
                with open_dict(gate_cfg):
                    gate_cfg.enabled = bool(flat_ablate["relational.gate.enabled"])
        spat_cfg = mem_cfg.get("spatial")
        if isinstance(spat_cfg, DictConfig) and "spatial.gate.enabled" in flat_ablate:
            gate_cfg = spat_cfg.get("gate")
            if isinstance(gate_cfg, DictConfig):
                with open_dict(gate_cfg):
                    gate_cfg.enabled = bool(flat_ablate["spatial.gate.enabled"])


__all__ = ["merge_memory_shortcuts", "apply_ablation_flags"]
