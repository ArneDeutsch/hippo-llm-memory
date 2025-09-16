# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
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
from transformers.pytorch_utils import Conv1D

from hippo_mem.adapters.lora import default_target_modules, export_adapter, merge_adapter
from hippo_mem.common import io
from hippo_mem.consolidation import ReplayDataset
from hippo_mem.testing.fake_hf import resolve_fake_model_id


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


def _format_records(records: Iterable[Dict[str, Any]]) -> tuple[list[str], list[Dict[str, Any]]]:
    """Return (texts, records) excluding empty prompt/answer pairs.

    Some replay stores may contain placeholder entries without meaningful
    ``prompt``/``answer`` fields.  Feeding such records to the tokenizer
    yields empty tensors which later trigger reshape errors inside the
    model.  We therefore skip blank pairs and keep ``texts`` aligned with
    the filtered ``records`` so downstream processing remains consistent.
    """

    texts: list[str] = []
    kept: list[Dict[str, Any]] = []
    for rec in records:
        prompt = rec.get("prompt") or rec.get("q") or ""
        answer = rec.get("answer") or rec.get("a") or ""
        text = f"{prompt}\n{answer}".strip()
        if text:
            texts.append(text)
            kept.append(rec)
    return texts, kept


def _compute_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Return KL(student‖teacher) for logit tensors."""

    s_log = torch.log_softmax(student, dim=-1)
    t_log = torch.log_softmax(teacher, dim=-1)
    return torch.nn.functional.kl_div(s_log, t_log, log_target=True, reduction="batchmean")


class ConsolidationTrainer:
    """Encapsulates model and tokenizer for consolidation training."""

    def __init__(self, model_name: str, seed: int) -> None:
        logging.info("loading model %s", model_name)
        resolved = resolve_fake_model_id(model_name) or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(resolved)
        torch.manual_seed(seed)
        self.model.train()
        self.lora_hash = ""

    def configure_lora(self, cfg: Dict[str, Any]) -> bool:
        modules = cfg.get("targets")
        if not modules or modules == "auto":
            modules = default_target_modules(self.model)
        else:
            available = {name.split(".")[-1] for name, _ in self.model.named_modules()}
            missing = [m for m in modules if m not in available]
            if missing:
                logging.warning("target modules %s not found; using defaults", missing)
                modules = default_target_modules(self.model)
        if not modules:
            logging.warning("no target modules for model")
            return False
        cfg["targets"] = modules
        fan_in_fan_out = any(
            isinstance(mod, Conv1D) and any(name.endswith(m) for m in modules)
            for name, mod in self.model.named_modules()
        )
        lora_cfg = LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            lora_dropout=cfg["dropout"],
            target_modules=modules,
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=fan_in_fan_out,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.lora_hash = hashlib.sha256(
            json.dumps(
                {
                    "rank": lora_cfg.r,
                    "alpha": lora_cfg.lora_alpha,
                    "dropout": lora_cfg.lora_dropout,
                    "targets": cfg["targets"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        logging.info("lora_config_hash=%s", self.lora_hash)
        return True

    def build_dataset(self, store_dir: str, session_id: str, cfg: Dict[str, Any]) -> ReplayDataset:
        return ReplayDataset(
            store_dir,
            session_id,
            policy=cfg.get("policy", "priority"),
            cycles=cfg.get("cycles"),
        )

    def _finalize(
        self, outdir: str, steps: int, lr: float, replay_samples: int, merge: bool
    ) -> Dict[str, Any]:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        if merge:
            merged = merge_adapter(self.model)
            merged.save_pretrained(outdir)
        else:
            export_adapter(self.model, outdir)
        self.tokenizer.save_pretrained(outdir)
        meta = {
            "steps": steps,
            "lr": lr,
            "replay_samples": replay_samples,
            "lora_config_hash": self.lora_hash,
            "merged": merge,
        }
        io.atomic_write_json(Path(outdir) / "meta.json", meta)
        return meta

    def training_loop(
        self,
        dataset: ReplayDataset,
        train_cfg: Dict[str, Any],
        kl_weight: float,
        outdir: str,
        merge: bool,
    ) -> Dict[str, Any]:
        iterator = iter(dataset)
        opt = torch.optim.AdamW(self.model.parameters(), lr=train_cfg["lr"])
        replay_count = 0
        for step in range(1, train_cfg["steps"] + 1):
            batch = [next(iterator, None) for _ in range(train_cfg["batch_size"])]
            batch = [b for b in batch if b is not None]
            if not batch:
                logging.info("empty batch; stopping")
                return self._finalize(outdir, step - 1, train_cfg["lr"], replay_count, merge)
            texts, batch = _format_records(batch)
            if not texts:
                continue
            toks = self.tokenizer(texts, return_tensors="pt", padding=True)
            # Some tokenizers may yield floating tensors; embeddings expect integer indices.
            toks = {k: v.long() for k, v in toks.items()}
            labels = toks["input_ids"].clone()
            out = self.model(**toks, labels=labels)
            loss = out.loss
            if kl_weight > 0.0:
                kls: List[torch.Tensor] = []
                for i, rec in enumerate(batch):
                    teacher = rec.get("teacher")
                    if teacher and "logits" in teacher:
                        t_logits = torch.tensor(teacher["logits"], dtype=out.logits.dtype)
                        s_logits = out.logits[i, -t_logits.size(0) :]
                        kls.append(_compute_kl(s_logits, t_logits))
                if kls:
                    loss = loss + kl_weight * torch.stack(kls).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            replay_count += len(batch)
            logging.info(
                "step=%d lr=%.2e loss=%.4f",
                step,
                train_cfg["lr"],
                loss.detach().item(),
            )
        return self._finalize(outdir, step, train_cfg["lr"], replay_count, merge)


def train(args: Args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute consolidation training and return metadata."""

    trainer = ConsolidationTrainer(args.model, args.seed)
    if not trainer.configure_lora(cfg["peft"]):
        return {}
    ds = trainer.build_dataset(args.store_dir, args.session_id, cfg["replay"])
    return trainer.training_loop(ds, cfg["train"], args.kl_weight, args.outdir, args.merge)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:  # pragma: no cover - CLI
    args = parse_args(argv)
    cfg = load_config(args.config)
    return train(args, cfg)


__all__ = [
    "Args",
    "load_config",
    "main",
    "ConsolidationTrainer",
    "parse_args",
    "train",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
