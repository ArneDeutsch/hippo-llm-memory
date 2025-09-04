"""Execution helpers for the evaluation harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import DictConfig, OmegaConf

from hippo_eval.datasets.loaders import load_dataset
from hippo_mem.common.gates import GateCounters
from hippo_mem.common.telemetry import gate_registry, registry
from hippo_mem.episodic.gating import WriteGate
from hippo_mem.relational.gating import RelationalGate
from hippo_mem.spatial.gating import SpatialGate


@dataclass
class Runner:
    """Container for evaluation configuration."""

    cfg: DictConfig


@dataclass
class RunResult:
    """Result from executing a suite."""

    rows: List[Dict[str, object]]
    metrics: Dict[str, object]
    flat_ablate: Dict[str, object]


def build_runner(cfg: DictConfig) -> Runner:
    """Return a :class:`Runner` with defaults applied to ``cfg``."""

    from hippo_eval.eval import harness as _h

    cfg = _h._apply_model_defaults(cfg)
    return Runner(cfg)


def run_suite(runner: Runner, suite: str | None = None) -> RunResult:
    """Execute ``suite`` using ``runner``'s configuration."""

    from hippo_eval.eval import harness as _h

    cfg = runner.cfg
    if suite is not None:
        cfg.suite = suite

    registry.reset()
    gate_registry.reset()
    gating: Dict[str, GateCounters] = {
        k: GateCounters() for k in ("episodic", "relational", "spatial")
    }

    base_cfg = OmegaConf.create(
        {
            "suite": cfg.suite,
            "n": cfg.n,
            "seed": cfg.seed,
            "model": cfg.model,
            "max_new_tokens": cfg.max_new_tokens,
            "use_chat_template": cfg.use_chat_template,
            "system_prompt": cfg.system_prompt,
            "pad_token_id": cfg.get("pad_token_id"),
            "eos_token_id": cfg.get("eos_token_id"),
        }
    )
    if cfg.get("preset"):
        preset_cfg = OmegaConf.load(cfg.preset)
        base_cfg = OmegaConf.merge(base_cfg, preset_cfg)

    dataset = _h._dataset_path(cfg.suite, cfg.n, cfg.seed, cfg.get("dataset_profile"))
    raw_tasks = load_dataset(dataset, {"n": cfg.n})
    tasks = [_h.Task(prompt=str(obj["prompt"]), answer=str(obj["answer"])) for obj in raw_tasks]
    flat_ablate = _h._flatten_ablate(base_cfg.get("ablate"))
    modules = _h._init_modules(
        base_cfg.get("memory"),
        flat_ablate,
        allow_dummy_stores=cfg.get("allow_dummy_stores", False),
    )
    if cfg.get("preset") and not _h.is_memory_preset(str(cfg.preset)):
        modules = {}
    if "episodic" in modules:
        modules["episodic"]["gate"] = WriteGate()
    if "relational" in modules:
        modules["relational"]["gate"] = RelationalGate()
    if "spatial" in modules:
        modules["spatial"]["gate"] = SpatialGate()

    model_id = (str(base_cfg.model) or "").strip()
    if not model_id:
        raise ValueError("cfg.model is empty. Pass --model or set $MODEL.")
    p = Path(model_id)
    if p.exists() and p.is_dir() and not (p / "config.json").exists():
        raise ValueError(
            f"Model path '{p}' exists but is not a Hugging Face model dir (missing config.json). "
            "Did you accidentally pass the repository root? Set --model correctly."
        )
    model_path = _h.to_absolute_path(model_id)
    tokenizer = _h.AutoTokenizer.from_pretrained(model_path)
    if cfg.get("pad_token_id") is not None:
        tokenizer.pad_token_id = cfg.pad_token_id
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.get("eos_token_id") is not None:
        tokenizer.eos_token_id = cfg.eos_token_id
    model = _h.AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if cfg.get("eos_token_id") is not None:
        model.generation_config.eos_token_id = cfg.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    retrieval_enabled = bool(base_cfg.get("retrieval", {}).get("enabled", False))
    long_ctx_enabled = bool(base_cfg.get("long_context", {}).get("enabled", False))

    rows, metrics, in_tokens, gen_tokens, elapsed = _h._evaluate(
        tasks,
        modules,
        tokenizer,
        model,
        int(base_cfg.max_new_tokens),
        use_chat_template=cfg.use_chat_template,
        system_prompt=cfg.system_prompt,
        suite=cfg.suite,
        retrieval_enabled=retrieval_enabled,
        long_context_enabled=long_ctx_enabled,
        compute_metrics=True,
        mode="test",
        gating=gating,
    )
    metrics["em"] = (
        metrics.get("em_norm", 0.0)
        if cfg.get("primary_em") == "norm"
        else metrics.get("em_raw", 0.0)
    )
    lat_mean = sum(r["latency_ms"] for r in rows) / max(1, len(rows))

    replay_samples = 0
    for _ in range(_h.get_replay_cycles(cfg)):
        replay_samples += _h._run_replay(base_cfg, modules, tasks)

    total_tokens = in_tokens + gen_tokens
    store_sizes, store_diags = _h._store_sizes(modules)
    retrieval_snaps = registry.all_snapshots()
    metrics_dict = {
        "version": 2,
        "phase": str(getattr(cfg, "mode", "test")),
        "suite": cfg.suite,
        "n": cfg.n,
        "seed": cfg.seed,
        "preset": cfg.get("preset"),
        "metrics": {
            cfg.suite: metrics,
            "compute": {
                "input_tokens": in_tokens,
                "generated_tokens": gen_tokens,
                "total_tokens": total_tokens,
                "time_ms_per_100": 100 * elapsed * 1000 / max(1, total_tokens),
                "rss_mb": _h._rss_mb(),
                "latency_ms_mean": lat_mean,
            },
        },
        "retrieval": retrieval_snaps,
        "gating": {k: asdict(v) for k, v in gating.items()},
        "replay": {"samples": replay_samples},
        "store": {
            "size": sum(store_sizes.values()),
            "per_memory": store_sizes,
            "diagnostics": store_diags,
        },
    }
    _h._enforce_guardrails(base_cfg, metrics, None, retrieval_snaps, has_memory=bool(modules))
    return RunResult(rows, metrics_dict, flat_ablate)
