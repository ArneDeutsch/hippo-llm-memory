"""Utilities for training LoRA adapters from replayed memory stores.

This module contains the implementation that was previously embedded in the
``scripts/replay_consolidate.py`` entry point.  Keeping the logic here allows
other modules to import and reuse the consolidation training helpers without
going through the script layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from hippo_mem.adapters.lora import default_target_modules, export_adapter, merge_adapter
from hippo_mem.common import io
from hippo_mem.consolidation import ReplayDataset


@dataclass
class Args:
    """CLI arguments for consolidation training."""

    store_dir: str
    session_id: str
    outdir: str
    model: str
    config: Optional[str]
    seed: int = 0
    kl_weight: float = 0.0
    merge: bool = False


def _parse_args(argv: Optional[List[str]] = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_dir", required=True)
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL_PATH"))
    parser.add_argument("--config", default=None, help="YAML file with training and LoRA settings")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument(
        "--merge", action="store_true", help="Merge adapter weights into the base model"
    )
    ns = parser.parse_args(argv)
    return Args(
        store_dir=ns.store_dir,
        session_id=ns.session_id,
        outdir=ns.outdir,
        model=ns.model,
        config=ns.config,
        seed=ns.seed,
        kl_weight=ns.kl_weight,
        merge=ns.merge,
    )


def parse_args(argv: Optional[List[str]] = None) -> Args:
    """Parse command line arguments into an :class:`Args` object."""

    return _parse_args(argv)


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``upd`` into ``base``."""

    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config with defaults for training and LoRA."""

    cfg: Dict[str, Any] = {
        "peft": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
        },
        "train": {"lr": 2.0e-4, "steps": 100, "batch_size": 4},
        "replay": {"policy": "priority", "cycles": 1},
    }
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _deep_update(cfg, data)
    return cfg


def _format_records(records: Iterable[Dict[str, Any]]) -> List[str]:
    """Convert replay records into text pairs suitable for tokenization."""

    texts = []
    for rec in records:
        prompt = rec.get("prompt", rec.get("q", ""))
        answer = rec.get("answer", rec.get("a", ""))
        texts.append(f"{prompt}\n{answer}".strip())
    return texts


def _compute_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Return KL(student‖teacher) for logit tensors."""

    s_log = torch.log_softmax(student, dim=-1)
    t_log = torch.log_softmax(teacher, dim=-1)
    return torch.nn.functional.kl_div(s_log, t_log, log_target=True, reduction="batchmean")


def train(args: Args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute consolidation training and return metadata."""

    logging.info("loading model %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    modules = cfg["peft"].get("targets")
    if not modules or modules == "auto":
        modules = default_target_modules(model)
    else:
        available = {name.split(".")[-1] for name, _ in model.named_modules()}
        missing = [m for m in modules if m not in available]
        if missing:
            logging.warning("target modules %s not found; using defaults", missing)
            modules = default_target_modules(model)
    if not modules:
        raise ValueError("Could not determine target modules for model")
    cfg["peft"]["targets"] = modules
    lora_cfg = LoraConfig(
        r=cfg["peft"]["rank"],
        lora_alpha=cfg["peft"]["alpha"],
        lora_dropout=cfg["peft"]["dropout"],
        target_modules=modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    lora_hash = hashlib.sha256(
        json.dumps(
            {
                "rank": lora_cfg.r,
                "alpha": lora_cfg.lora_alpha,
                "dropout": lora_cfg.lora_dropout,
                "targets": cfg["peft"]["targets"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    logging.info("lora_config_hash=%s", lora_hash)

    torch.manual_seed(args.seed)
    model.train()
    ds = ReplayDataset(
        args.store_dir,
        args.session_id,
        policy=cfg["replay"].get("policy", "priority"),
        cycles=cfg["replay"].get("cycles"),
    )
    iterator = iter(ds)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    replay_count = 0
    for step in range(1, cfg["train"]["steps"] + 1):
        batch = list(next(iterator, None) for _ in range(cfg["train"]["batch_size"]))
        batch = [b for b in batch if b is not None]
        if not batch:
            break
        texts = _format_records(batch)
        toks = tokenizer(texts, return_tensors="pt", padding=True)
        labels = toks["input_ids"].clone()
        out = model(**toks, labels=labels)
        loss = out.loss
        if args.kl_weight > 0.0:
            kls: List[torch.Tensor] = []
            for i, rec in enumerate(batch):
                teacher = rec.get("teacher")
                if teacher and "logits" in teacher:
                    t_logits = torch.tensor(teacher["logits"], dtype=out.logits.dtype)
                    s_logits = out.logits[i, -t_logits.size(0) :]
                    kls.append(_compute_kl(s_logits, t_logits))
            if kls:
                loss = loss + args.kl_weight * torch.stack(kls).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        replay_count += len(batch)
        logging.info("step=%d lr=%.2e loss=%.4f", step, cfg["train"]["lr"], float(loss))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.merge:
        merged = merge_adapter(model)
        merged.save_pretrained(args.outdir)
    else:
        export_adapter(model, args.outdir)
    tokenizer.save_pretrained(args.outdir)
    meta = {
        "steps": step,
        "lr": cfg["train"]["lr"],
        "replay_samples": replay_count,
        "lora_config_hash": lora_hash,
        "merged": args.merge,
    }
    io.atomic_write_json(Path(args.outdir) / "meta.json", meta)
    return meta


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:  # pragma: no cover - CLI
    args = parse_args(argv)
    cfg = load_config(args.config)
    return train(args, cfg)


__all__ = [
    "Args",
    "load_config",
    "main",
    "parse_args",
    "train",
]
