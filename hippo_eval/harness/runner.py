"""Execution helpers for the evaluation harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    """Execute evaluation suites."""

    cfg: DictConfig

    def prepare(
        self, suite: str | None = None
    ) -> Tuple[DictConfig, List, Dict, Dict[str, GateCounters], Dict[str, object]]:
        """Prepare config, tasks and modules."""

        from hippo_eval.eval import harness as _h

        if suite is not None:
            self.cfg.suite = suite

        registry.reset()
        gate_registry.reset()
        gating: Dict[str, GateCounters] = {
            k: GateCounters() for k in ("episodic", "relational", "spatial")
        }

        base_cfg = OmegaConf.create(
            {
                "suite": self.cfg.suite,
                "n": self.cfg.n,
                "seed": self.cfg.seed,
                "model": self.cfg.model,
                "max_new_tokens": self.cfg.max_new_tokens,
                "use_chat_template": self.cfg.use_chat_template,
                "system_prompt": self.cfg.system_prompt,
                "pad_token_id": self.cfg.get("pad_token_id"),
                "eos_token_id": self.cfg.get("eos_token_id"),
            }
        )
        if self.cfg.get("preset"):
            preset_cfg = OmegaConf.load(self.cfg.preset)
            base_cfg = OmegaConf.merge(base_cfg, preset_cfg)

        dataset = _h._dataset_path(
            self.cfg.suite, self.cfg.n, self.cfg.seed, self.cfg.get("dataset_profile")
        )
        raw_tasks = load_dataset(dataset, {"n": self.cfg.n})
        tasks = [_h.Task(prompt=str(obj["prompt"]), answer=str(obj["answer"])) for obj in raw_tasks]

        flat_ablate = _h._flatten_ablate(base_cfg.get("ablate"))
        modules = _h._init_modules(
            base_cfg.get("memory"),
            flat_ablate,
            allow_dummy_stores=self.cfg.get("allow_dummy_stores", False),
        )
        if self.cfg.get("preset") and not _h.is_memory_preset(str(self.cfg.preset)):
            modules = {}

        if not modules:
            return base_cfg, tasks, modules, gating, flat_ablate

        if "episodic" in modules:
            modules["episodic"]["gate"] = WriteGate()
        if "relational" in modules:
            modules["relational"]["gate"] = RelationalGate()
        if "spatial" in modules:
            modules["spatial"]["gate"] = SpatialGate()

        return base_cfg, tasks, modules, gating, flat_ablate

    def load_model_and_tokenizer(self, base_cfg: DictConfig) -> Tuple[object, object]:
        """Load tokenizer and model, validating paths."""

        from hippo_eval.eval import harness as _h

        model_id = (str(base_cfg.model) or "").strip()
        if not model_id:
            raise ValueError("cfg.model is empty. Pass --model or set $MODEL.")
        p = Path(model_id)
        if p.exists() and p.is_dir() and not (p / "config.json").exists():
            msg = (
                f"Model path '{p}' exists but is not a Hugging Face model dir (missing config.json). "
                "Did you accidentally pass the repository root? Set --model correctly."
            )
            raise ValueError(msg)

        model_path = _h.to_absolute_path(model_id)
        tokenizer = _h.AutoTokenizer.from_pretrained(model_path)
        if self.cfg.get("pad_token_id") is not None:
            tokenizer.pad_token_id = self.cfg.pad_token_id
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if self.cfg.get("eos_token_id") is not None:
            tokenizer.eos_token_id = self.cfg.eos_token_id

        model = _h.AutoModelForCausalLM.from_pretrained(model_path)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if self.cfg.get("eos_token_id") is not None:
            model.generation_config.eos_token_id = self.cfg.eos_token_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model

    def evaluate(
        self,
        base_cfg: DictConfig,
        tasks: List,
        modules: Dict,
        gating: Dict[str, GateCounters],
        tokenizer: object,
        model: object,
    ) -> Tuple[List[Dict[str, object]], Dict[str, float], int, int, float, float]:
        """Run the evaluation."""

        from hippo_eval.eval import harness as _h

        retrieval_enabled = bool(base_cfg.get("retrieval", {}).get("enabled", False))
        long_ctx_enabled = bool(base_cfg.get("long_context", {}).get("enabled", False))

        rows, metrics, in_tokens, gen_tokens, elapsed = _h._evaluate(
            tasks,
            modules,
            tokenizer,
            model,
            int(base_cfg.max_new_tokens),
            use_chat_template=self.cfg.use_chat_template,
            system_prompt=self.cfg.system_prompt,
            suite=self.cfg.suite,
            retrieval_enabled=retrieval_enabled,
            long_context_enabled=long_ctx_enabled,
            compute_metrics=True,
            mode="test",
            gating=gating,
        )
        metrics["em"] = (
            metrics.get("em_norm", 0.0)
            if self.cfg.get("primary_em") == "norm"
            else metrics.get("em_raw", 0.0)
        )
        lat_mean = sum(r["latency_ms"] for r in rows) / max(1, len(rows))
        return rows, metrics, in_tokens, gen_tokens, elapsed, lat_mean

    def run_replay_cycles(self, base_cfg: DictConfig, modules: Dict, tasks: List) -> int:
        """Replay memory samples when modules are present."""

        from hippo_eval.eval import harness as _h

        if not modules:
            return 0

        replay_samples = 0
        for _ in range(_h.get_replay_cycles(self.cfg)):
            replay_samples += _h._run_replay(base_cfg, modules, tasks)
        return replay_samples

    def run(self, suite: str | None = None) -> RunResult:
        """Execute ``suite`` using the runner's configuration."""

        from hippo_eval.eval import harness as _h

        base_cfg, tasks, modules, gating, flat_ablate = self.prepare(suite)
        tokenizer, model = self.load_model_and_tokenizer(base_cfg)
        rows, metrics, in_tokens, gen_tokens, elapsed, lat_mean = self.evaluate(
            base_cfg, tasks, modules, gating, tokenizer, model
        )
        replay_samples = self.run_replay_cycles(base_cfg, modules, tasks)

        total_tokens = in_tokens + gen_tokens
        store_sizes, store_diags = _h._store_sizes(modules)
        retrieval_snaps = registry.all_snapshots()
        metrics_dict = {
            "version": 2,
            "phase": str(getattr(self.cfg, "mode", "test")),
            "suite": self.cfg.suite,
            "n": self.cfg.n,
            "seed": self.cfg.seed,
            "preset": self.cfg.get("preset"),
            "metrics": {
                self.cfg.suite: metrics,
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
    """Convenience wrapper to run a suite with ``runner``."""

    return runner.run(suite)
