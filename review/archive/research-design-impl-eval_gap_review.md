# Research ↔ Design ↔ Implementation ↔ Evaluation Gap Review

**Generated:** 2025-08-25 11:33

## Scope
This review aligns the *research/experiment-synthesis.md* (ground truth) with the current **design**, **implementation**, and **validation**. It focuses on hippocampus-inspired capabilities: episodic/relational/spatial memory, gating, replay-driven consolidation, and (crucially) **long-term consolidation across conversations/sessions**.

## Research document signals (selected)

> positional encoding variants (RoPE, relative, ALiBi), feed-forward blocks with gated activations (SwiGLU/GEGLU), residual connections and (pre-)LayerNorm. It expla
> The paper characterizes the hippocampus as a rapid, sparse, content-addressable episodic memory system embedded in the trisynaptic circuit (EC→DG→CA3→CA1). DG performs
> ian plasticity (LTP) modulated by neuromodulators (DA/NE) with novelty/salience gating. Consolidation proceeds from synaptic stabilization to systems-level replay dur
> nd maps hippocampal principles to ML (CLS, Hopfield attractors, SDM, generative replay, prosthetic MIMO models).       &#x20;  ### Key Terms & Definitions  * **Trisyn
> synapses index cortical representations; content-addressable.&#x20; * **Systems consolidation:** hippocampal-to-cortical transfer via SWR replay during SWS; CA2 may coordina
> ple timing. &#x20; * **CLS (Complementary Learning Systems):** fast hippocampal episodic vs. slow cortical semantic learning.&#x20; * **Neuromodulatory gating:** DA/NE

## Current design & plans (files detected)

- **DESIGN**: hippo-llm-memory-main/DESIGN.md
- **PROJECT_PLAN**: hippo-llm-memory-main/PROJECT_PLAN.md
- **EVAL_PLAN**: hippo-llm-memory-main/EVAL_PLAN.md
- **M9_PLAN**: hippo-llm-memory-main/MILESTONE_9_PLAN.md

### Notable evaluation presets

- hippo-llm-memory-main/configs/eval/default.yaml → retrieval=False, replay=0, gating=False, longctx=False, chat=False
- hippo-llm-memory-main/configs/eval/baselines/core.yaml → retrieval=False, replay=0, gating=False, longctx=False, chat=False
- hippo-llm-memory-main/configs/eval/baselines/longctx.yaml → retrieval=False, replay=0, gating=False, longctx=True, chat=False
- hippo-llm-memory-main/configs/eval/baselines/rag.yaml → retrieval=True, replay=0, gating=False, longctx=False, chat=False
- hippo-llm-memory-main/configs/eval/memory/all.yaml → retrieval=False, replay=0, gating=False, longctx=False, chat=False
- hippo-llm-memory-main/configs/eval/memory/hei_nw.yaml → retrieval=True, replay=1, gating=False, longctx=False, chat=False
- hippo-llm-memory-main/configs/eval/memory/sgc_rss.yaml → retrieval=True, replay=1, gating=False, longctx=False, chat=False
- hippo-llm-memory-main/configs/eval/memory/smpd.yaml → retrieval=True, replay=1, gating=False, longctx=False, chat=False

## Findings — Gaps and impact

### F. Missing RAG baseline
A non-learning RAG baseline helps compare learned memory vs retrieval-only systems.

### G. Decoding not matched to metrics
Exact-match EM requires terse span outputs. Current defaults (chat template + long max_new_tokens) cause refusals and verbose text, deflating EM/F1.

### H. Acceptance criteria too weak
Runs can silently disable memory features. We need preflight/resolved-config dumps and post-run assertions (e.g., retrieval_requests > 0) for memory presets.

## Remediation plan

### 1) Cross-run persistence & two-phase protocol
- Add a persistence interface for all stores: `save(dir, session_id)` / `load(dir, session_id)` with JSONL or parquet backends.
- CLI: `--store_dir`, `--session_id`, `--persist=true`.
- Protocol:
  1. **Teach run** (`mode=teach`): present facts; gates+writes enabled; results not graded; saves stores.
  2. **Sleep/Replay run** (`mode=replay`): optional background replay over saved stores; write to cortical cache (e.g., a distilled LM or KV index).
  3. **Test run** (`mode=test`): new process; load stores; query without re-presenting facts; grade EM/F1.
- Add acceptance checks: test answers remain available after process restart; measure *delayed recall* and *retention over interference*.

### 2) Explicit replay policy
- Implement a scheduler: uniform/priority-based sampling, spaced intervals, and noise injection.
- Config knobs: `replay.policy={uniform,priority,spaced}`, `replay.rate`, `replay.noise_level`, `replay.max_items`.
- Metrics: replay count, items rehearsed, pre→post EM/F1 delta, forgetting curves.

### 3) Gate instrumentation & tests
- Telemetry: count write attempts, gate accept/reject, and retrieved-before/after gating.
- Stress tests: inject distractors; verify gates maintain precision/recall.
- Acceptance: in a distractor-heavy set, gated writes improve downstream EM/F1 vs ungated.

### 4) Relational & spatial task suites
- Provide minimal synthetic suites mirroring episodic format (n=50):
  - **Relational:** subject–predicate–object queries, compositional chains.
  - **Spatial:** grid/world coordinates, path queries, proximity.
- Report same metrics; ensure parity of harness flags and telemetry.

### 5) Add ablation baselines
- **Long-context concat** preset (no retrieval): append supports to prompt.
- **Simple RAG** preset: nearest-neighbor over saved snippets (no learning).
- Use alongside memory presets to isolate where gains originate.

### 6) Span-short decoding profile
- New baseline `baselines/span_short` with chat template ON, short span system prompt, `max_new_tokens<=8`.
- Apply the same decoding profile to memory presets for fair comparisons on EM.

### 7) Stronger acceptance/CI
- Preflight: dump resolved config at start of each run.
- Post-run assertions:
  - For memory presets: `retrieval.requests > 0`, `replay.cycles >= 1` (if configured).
  - Refusal-rate guard: fail if >50% predictions match refusal regexes.
- Unit tests for preset schemas and store save/load.

## Milestone updates

**Option 1 — Insert Milestone 9.5: “Cross‑Session Consolidation” (recommended)**
- **Objective:** Demonstrate memory that persists across runs and improves delayed recall.
- **Deliverables:**
  1. Store persistence API + CLI (`--store_dir`, `--session_id`, `--persist`).
  2. Two-phase harness (`mode=teach|replay|test`) with scripts.
  3. New eval suites for relational/spatial (n=50 each).
  4. Baselines: `longctx`, `rag`, `span_short`.
  5. CI: pre/post-run assertions + refusal-rate guard.
- **Success criteria:** On `episodic@50`, *delayed* EM improves by ≥+0.20 absolute vs core baseline; retrieval telemetry shows activity; across-process retention confirmed.

**Option 2 — Expand Milestone 9**
- Integrate items (1–5) directly into M9; adjust timelines accordingly.
- Use your current runs as M9 “intra-session” evidence, and add M9.2 for “inter-session” consolidation.

## Codex task list

1. [stores] Implement `save/load` for episodic/relational/spatial stores (JSONL backend) and thread-safe open/close.
2. [cli] Add `--store_dir`, `--session_id`, `--persist`, and `--mode={teach,replay,test}` to `scripts/eval_model.py`.
3. [harness] Teach/Test pipeline: in teach mode read facts only; in test mode load stores and evaluate without facts.
4. [replay] Add scheduler policies and knobs; default `replay.policy=uniform`, `replay.cycles=1`.
5. [telemetry] Log retrieval requests, gate attempts, write accepts/rejects; flush to `metrics.json`.
6. [eval] Create `tasks/relational_50.jsonl` and `tasks/spatial_50.jsonl` + loaders; mirror episodic formats.
7. [presets] Add `baselines/longctx.yaml`, `baselines/rag.yaml`, `baselines/span_short.yaml`, and `memory/hei_nw_span.yaml`.
8. [tests] Unit tests: preset schema, refusal-rate guard, store save/load round-trip, replay policy sampling.
9. [ci] Fail memory runs if `retrieval.requests==0` or refusal rate > 0.5.
10. [docs] Update README, PROJECT_PLAN, EVAL_PLAN, and MILESTONE_9_PLAN (or create MILESTONE_9_5_PLAN.md).

## Appendix — File references

- Research: hippo-llm-memory-main/research/experiment-synthesis.md
- Design: hippo-llm-memory-main/DESIGN.md
- Project plan: hippo-llm-memory-main/PROJECT_PLAN.md
- Eval plan: hippo-llm-memory-main/EVAL_PLAN.md
- Milestone 9 plan: hippo-llm-memory-main/MILESTONE_9_PLAN.md

### Notable implementation files (first 20 heads)

**hippo-llm-memory-main/src/__init__.py**
```

```

**hippo-llm-memory-main/src/eval/__init__.py**
```

```

**hippo-llm-memory-main/src/eval/encode.py**
```
"""Prompt encoding utilities for evaluation harness.

Provides :func:`encode_prompt` which applies chat templates when
available and falls back to direct tokenization otherwise.
"""

from __future__ import annotations

from typing import Dict

import torch


def encode_prompt(
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    use_chat_template: bool = True,
    system_prompt: str = (
        "Answer with the exact shortest span from the prompt. No explanations."
    ),
) -> Dict[str, torch.Tensor]:
    """Return ``input_ids`` for ``prompt`` on ``device``.

    If ``use_chat_template`` is ``True`` and ``tokenizer`` supports
    :func:`~transformers.PreTrainedTokenizer.apply_chat_template` with a defined
    ``chat_template``, the prompt is wrapped in a simple system/user dialogue and
    encoded via ``apply_chat_template``. Otherwise the prompt is tokenized
    directly.
    """

    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    )
    if use_chat_template and has_chat:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return {"input_ids": input_ids.to(device)}
```

**hippo-llm-memory-main/src/eval/models.py**
```
"""Model registry loader for evaluation harness.

Reads ``configs/models.yaml`` and returns configuration for a
specific ``model_id``.  Registry entries may specify whether chat
templates should be applied and provide generation defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULTS: Dict[str, Any] = {
    "use_chat_template": False,
    "system_prompt": "Answer with the exact shortest span from the prompt. No explanations.",
    "eos_token_id": None,
    "pad_token_id": None,
    "max_new_tokens": 32,
}


def load_model_config(model_id: str, path: str | Path = "configs/models.yaml") -> Dict[str, Any]:
    """Return registry settings for ``model_id``.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier used as key in ``models.yaml``.
    path:
        Location of the registry file relative to the repository root.
    """

    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    defaults = {**_DEFAULTS, **data.get("defaults", {})}
    model_cfg = data.get(model_id, {})
    merged = {**defaults, **model_cfg}
    return merged
```
